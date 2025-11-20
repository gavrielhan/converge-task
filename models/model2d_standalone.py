#!/usr/bin/env python3
"""
Standalone script for Model 2D: D-SCRIPT (Official GitHub Implementation)
==========================================================================

This script uses the official D-SCRIPT implementation from GitHub
to predict protein-protein interactions directly from sequences.

Reference: https://github.com/samsledje/D-SCRIPT
"""

from __future__ import annotations

import argparse
import json
import pickle
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Try to import dscript
try:
    from dscript.pretrained import get_pretrained
    from dscript.language_model import lm_embed
    DSCRIPT_AVAILABLE = True
except ImportError:
    DSCRIPT_AVAILABLE = False

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
ESM_CACHE_PATH = PROJECT_ROOT / "cache/esm_embeddings.pkl"
RESULTS_DIR = PROJECT_ROOT
PLOT_DIR = PROJECT_ROOT / "plot"
PLOT_DIR.mkdir(exist_ok=True)


def load_curated_data():
    """Load pre-computed ESM-2 features and labels from curated_data/."""
    if not CURATED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Directory {CURATED_DATA_DIR} not found. "
            "Please run prepare_dataset.py first to generate curated data."
        )
    
    # Load metadata
    with (CURATED_DATA_DIR / "split_info.json").open('r') as f:
        metadata = json.load(f)
    
    # Load labels
    y_train = np.load(CURATED_DATA_DIR / "train_labels.npy")
    y_test = np.load(CURATED_DATA_DIR / "test_labels.npy")
    
    # Load pairs
    with (CURATED_DATA_DIR / "train_pairs.pkl").open('rb') as f:
        train_pairs = pickle.load(f)
    with (CURATED_DATA_DIR / "test_pairs.pkl").open('rb') as f:
        test_pairs = pickle.load(f)
    
    return {
        'metadata': metadata,
        'y_train': y_train,
        'y_test': y_test,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
    }


def load_dscript_model(device: str = "cpu", model_version: str = "human_v2"):
    """Load pretrained D-SCRIPT model.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
        model_version: Model version ('human_v1', 'human_v2', 'human_tt3d', or 'lm_v1')
    
    Returns:
        D-SCRIPT interaction model
    """
    if not DSCRIPT_AVAILABLE:
        raise ImportError(
            "dscript library not installed. "
            "Install with: pip install dscript\n"
            "Or from GitHub: pip install git+https://github.com/samsledje/D-SCRIPT.git"
        )
    
    print("Loading D-SCRIPT model...")
    print(f"  Device: {device}")
    print(f"  Model version: {model_version}")
    
    # Load pretrained interaction model
    model = get_pretrained(model_version)
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    model.use_cuda = use_cuda
    if use_cuda:
        model = model.cuda()
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def predict_interaction(
    seq_a: str,
    seq_b: str,
    model,
    device: str = "cpu"
) -> float:
    """
    Predict interaction probability for a pair of protein sequences.
    
    Args:
        seq_a: First protein sequence
        seq_b: Second protein sequence
        model: D-SCRIPT interaction model
        device: Device to run on
    
    Returns:
        Probability of interaction (0-1)
    """
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    
    try:
        with torch.no_grad():
            # Embed sequences using language model
            # lm_embed already returns batched tensors of shape (1, seq_len, embedding_dim)
            z0 = lm_embed(seq_a, use_cuda=use_cuda)
            z1 = lm_embed(seq_b, use_cuda=use_cuda)
            
            # Ensure tensors are on the correct device
            # lm_embed may not respect use_cuda parameter correctly
            if use_cuda:
                if not z0.is_cuda:
                    z0 = z0.cuda()
                if not z1.is_cuda:
                    z1 = z1.cuda()
            else:
                if z0.is_cuda:
                    z0 = z0.cpu()
                if z1.is_cuda:
                    z1 = z1.cpu()
            
            # Ensure model is on the same device as inputs
            model_device = next(model.parameters()).device
            if z0.device != model_device:
                z0 = z0.to(model_device)
                z1 = z1.to(model_device)
            
            # Predict interaction
            # model.predict expects (batch, seq_len, embedding_dim) for both inputs
            # It should handle different sequence lengths (N and M) to create N×M contact map
            prob = model.predict(z0, z1)
            
            # Extract probability (model already applies sigmoid)
            if isinstance(prob, tuple):
                prob = prob[0]  # predict returns (prob, contact_map)
            
            # prob is a tensor, extract the scalar value
            if isinstance(prob, torch.Tensor):
                return prob.item() if prob.numel() == 1 else float(prob[0])
            return float(prob)
    except RuntimeError as e:
        # Re-raise with more context
        raise RuntimeError(f"Error predicting interaction for sequences of lengths {len(seq_a)} and {len(seq_b)}: {e}")


def predict_batch(
    pairs: List[Tuple[str, str, str, str]],
    model,
    device: str = "cpu",
    batch_size: int = 32,
    verbose: bool = True
) -> np.ndarray:
    """
    Predict interactions for a batch of protein pairs.
    
    Args:
        pairs: List of (id_a, id_b, seq_a, seq_b) tuples
        model: D-SCRIPT interaction model
        device: Device to run on
        batch_size: Batch size for processing
        verbose: Whether to print progress
    
    Returns:
        Array of interaction probabilities
    """
    n_pairs = len(pairs)
    predictions = np.zeros(n_pairs, dtype=np.float32)
    
    if verbose:
        print(f"  Predicting {n_pairs} pairs...")
        print(f"    Progress: 0/{n_pairs} (0.0%)", end='', flush=True)
    
    # Process pairs sequentially for memory efficiency
    errors = 0
    for i, (id_a, id_b, seq_a, seq_b) in enumerate(pairs):
        try:
            prob = predict_interaction(seq_a, seq_b, model, device)
            predictions[i] = prob
        except Exception as e:
            errors += 1
            if verbose:
                print(f"\n    Warning: Error predicting pair ({id_a}, {id_b}): {e}")
            # Default to 0.5 if prediction fails
            predictions[i] = 0.5
        
        # Update progress more frequently
        if verbose:
            progress = i + 1
            percentage = (progress / n_pairs) * 100
            # Update every 10 pairs, or at milestones (10%, 25%, 50%, 75%, 100%)
            if progress % 10 == 0 or progress in [int(n_pairs * 0.1), int(n_pairs * 0.25), 
                                                   int(n_pairs * 0.5), int(n_pairs * 0.75), n_pairs]:
                print(f"\r    Progress: {progress}/{n_pairs} ({percentage:.1f}%)", end='', flush=True)
    
    if verbose:
        print()  # New line after progress
        if errors > 0:
            print(f"  ⚠ {errors} pairs had errors (using default 0.5)")
        print(f"  ✓ Completed predictions for {n_pairs} pairs")
    
    return predictions


def evaluate_predictions(test_pairs, y_test, predictions):
    """Evaluate predictions against ground truth."""
    print(f"\n{'='*60}")
    print("EVALUATING PREDICTIONS")
    print(f"{'='*60}")
    
    n_samples = len(test_pairs)
    all_probs = np.array(predictions, dtype=np.float32)
    all_labels = np.array(y_test, dtype=np.int32)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n  Test samples: {len(all_labels)}")
    print(f"  Positive samples: {np.sum(all_labels)}")
    print(f"  Negative samples: {np.sum(1 - all_labels)}")
    
    print(f"\n  METRICS:")
    print(f"    Accuracy:   {accuracy:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"    F1-Score:   {f1:.4f}")
    print(f"    ROC-AUC:    {roc_auc:.4f}")
    print(f"    PR-AUC:     {pr_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    [[{cm[0, 0]:4d} {cm[0, 1]:4d}]")
    print(f"     [{cm[1, 0]:4d} {cm[1, 1]:4d}]]")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone D-SCRIPT Model 2D - Official GitHub Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (test set only)
  python model2d_standalone.py

  # Predict on both train and test sets
  python model2d_standalone.py --predict-train

  # Use GPU
  python model2d_standalone.py --device cuda

Note:
  Install D-SCRIPT with: pip install dscript
  Or from GitHub: pip install git+https://github.com/samsledje/D-SCRIPT.git
  
  If automatic download fails, download pretrained/dscript.pt from:
  https://github.com/samsledje/D-SCRIPT/tree/main/pretrained
        """
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to use (default: cpu)")
    parser.add_argument("--predict-train", action="store_true",
                       help="Also predict on training set")
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL 2D: D-SCRIPT (Official GitHub Implementation)")
    print("="*60)
    print(f"Device: {args.device}")
    
    # Check dscript availability
    if not DSCRIPT_AVAILABLE:
        print("\n✗ dscript library not installed.")
        print("  Install with: pip install dscript")
        print("  Or from GitHub: pip install git+https://github.com/samsledje/D-SCRIPT.git")
        return 1
    
    # Load data
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    try:
        data = load_curated_data()
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        traceback.print_exc()
        return 1
    
    metadata = data['metadata']
    y_train = data['y_train']
    y_test = data['y_test']
    train_pairs = data['train_pairs']
    test_pairs = data['test_pairs']
    
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    print(f"  Train positive: {np.sum(y_train)}")
    print(f"  Test positive: {np.sum(y_test)}")
    
    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    try:
        model = load_dscript_model(args.device)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        traceback.print_exc()
        return 1
    
    # Predict on test set
    print(f"\n{'='*60}")
    print("PREDICTING ON TEST SET")
    print(f"{'='*60}")
    try:
        test_predictions = predict_batch(
            test_pairs, model, args.device, verbose=True
        )
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        traceback.print_exc()
        return 1
    
    # Evaluate test set
    test_metrics = evaluate_predictions(test_pairs, y_test, test_predictions)
    
    # Optionally predict on train set
    train_metrics = None
    if args.predict_train:
        print(f"\n{'='*60}")
        print("PREDICTING ON TRAIN SET")
        print(f"{'='*60}")
        try:
            train_predictions = predict_batch(
                train_pairs, model, args.device, verbose=True
            )
            train_metrics = evaluate_predictions(train_pairs, y_train, train_predictions)
        except Exception as e:
            print(f"\n✗ Error during train prediction: {e}")
            traceback.print_exc()
            return 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  TEST SET:")
    print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"    F1-Score: {test_metrics['f1']:.4f}")
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    
    if train_metrics:
        print(f"\n  TRAIN SET:")
        print(f"    ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"    F1-Score: {train_metrics['f1']:.4f}")
        print(f"    Accuracy: {train_metrics['accuracy']:.4f}")
    
    print("\n✓ D-SCRIPT workflow completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
