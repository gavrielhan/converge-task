#!/usr/bin/env python3
"""
Model 2: Improved Neural Network Variants for PPI Prediction

Implements three improved variants:
- Model 2A: Better MLP (GELU, LayerNorm, wider layers)
- Model 2B: Siamese MLP architecture
- Model 2C: Transformer-classifier on ESM-2 embeddings

Supports K-fold cross-validation for robust evaluation.

Run prepare_dataset.py first to generate curated_data/ with ESM-2 features.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Try to import transformers for Model 2C
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
CACHE_DIR = PROJECT_ROOT / "cache"
ESM_CACHE_PATH = CACHE_DIR / "esm_embeddings.pkl"
RESULTS_PATH = PROJECT_ROOT / "model2_results.txt"
PLOT_DIR = PROJECT_ROOT / "plot"
PLOT_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "model2"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class Model2A_ImprovedMLP(nn.Module):
    """
    Model 2A: Better MLP with GELU, LayerNorm, wider layers.
    
    Architecture:
    - Input: 5120 dims (pair features)
    - Dense(2048) -> GELU -> LayerNorm -> Dropout(0.3)
    - Dense(512) -> GELU -> Dropout(0.2)
    - Dense(128) -> GELU -> Dropout(0.1)
    - Dense(1) -> Sigmoid
    """
    
    def __init__(self, input_dim: int = 5120):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.ln1 = nn.LayerNorm(2048)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(2048, 512)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class Model2B_SiameseMLP(nn.Module):
    """
    Model 2B: Siamese MLP architecture.
    
    Architecture:
    - Each protein: emb -> Dense(1024) -> GELU -> Dense(256) -> GELU -> h
    - Pair: concat(hA, hB, |hA-hB|, hA*hB)
    - Dense(256) -> GELU -> Dense(64) -> GELU -> Dense(1) -> Sigmoid
    """
    
    def __init__(self, protein_emb_dim: int = 1280):
        super().__init__()
        # Siamese branches (shared weights)
        self.protein_fc1 = nn.Linear(protein_emb_dim, 1024)
        self.protein_fc2 = nn.Linear(1024, 256)
        
        # Pair combination
        pair_dim = 256 * 4  # concat, diff, product
        self.pair_fc1 = nn.Linear(pair_dim, 256)
        self.pair_fc2 = nn.Linear(256, 64)
        self.pair_fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, emb_a, emb_b):
        # Process each protein independently
        h_a = self.protein_fc1(emb_a)
        h_a = F.gelu(h_a)
        h_a = self.protein_fc2(h_a)
        h_a = F.gelu(h_a)
        
        h_b = self.protein_fc1(emb_b)
        h_b = F.gelu(h_b)
        h_b = self.protein_fc2(h_b)
        h_b = F.gelu(h_b)
        
        # Combine pair features
        diff = torch.abs(h_a - h_b)
        product = h_a * h_b
        pair = torch.cat([h_a, h_b, diff, product], dim=1)
        
        # Final layers
        x = self.pair_fc1(pair)
        x = F.gelu(x)
        x = self.pair_fc2(x)
        x = F.gelu(x)
        x = self.pair_fc3(x)
        x = self.sigmoid(x)
        return x


class Model2C_TransformerClassifier(nn.Module):
    """
    Model 2C: Transformer encoder on ESM-2 embeddings.
    
    Uses a lightweight transformer to encode each protein embedding,
    then combines them for pair classification.
    """
    
    def __init__(self, protein_emb_dim: int = 1280, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        # Project embeddings to transformer dimension
        self.proj = nn.Linear(protein_emb_dim, hidden_dim)
        
        # Transformer encoder (single head for efficiency)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pair combination and classification
        pair_dim = hidden_dim * 4
        self.fc1 = nn.Linear(pair_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, emb_a, emb_b):
        # Project embeddings
        h_a = self.proj(emb_a).unsqueeze(1)  # Add sequence dimension
        h_b = self.proj(emb_b).unsqueeze(1)
        
        # Encode with transformer
        h_a = self.transformer(h_a).squeeze(1)
        h_b = self.transformer(h_b).squeeze(1)
        
        # Combine pair features
        diff = torch.abs(h_a - h_b)
        product = h_a * h_b
        pair = torch.cat([h_a, h_b, diff, product], dim=1)
        
        # Classify
        x = self.fc1(pair)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_cache(path: Path) -> Dict:
    """Load feature cache from disk."""
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    return {}


def get_available_folds():
    """Get list of available fold directories."""
    if not CURATED_DATA_DIR.exists():
        return []
    
    folds = []
    for i in range(10):  # Check up to 10 folds
        fold_dir = CURATED_DATA_DIR / f"fold_{i}"
        if fold_dir.exists() and (fold_dir / "train_labels.npy").exists():
            folds.append(i)
    
    return sorted(folds)


def load_curated_data(fold_idx: int = None):
    """
    Load pre-computed ESM-2 features and labels from curated_data/.
    
    Args:
        fold_idx: If provided, load data from fold_{fold_idx}/ directory.
                 If None, try to load from root (backward compatibility).
    
    Returns:
        Dictionary with data and metadata
    """
    if not CURATED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Directory {CURATED_DATA_DIR} not found. "
            "Please run prepare_dataset.py first to generate curated data."
        )
    
    # Determine data directory
    if fold_idx is not None:
        fold_dir = CURATED_DATA_DIR / f"fold_{fold_idx}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory {fold_dir} not found.")
        data_dir = fold_dir
        metadata_path = fold_dir / "split_info.json"
    else:
        # Try root directory (backward compatibility)
        data_dir = CURATED_DATA_DIR
        metadata_path = CURATED_DATA_DIR / "split_info.json"
        if not metadata_path.exists():
            # Try global metadata
            metadata_path = CURATED_DATA_DIR / "global_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    # Load metadata
    with metadata_path.open('r') as f:
        metadata = json.load(f)
    
    if not metadata.get("esm2_available", False):
        raise ValueError(
            "ESM-2 features not available in curated data. "
            "Run prepare_dataset.py without --skip-esm to generate them."
        )
    
    # Load labels
    y_train = np.load(data_dir / "train_labels.npy")
    y_test = np.load(data_dir / "test_labels.npy")
    
    # Load pair features (for Model 2A)
    X_train_pair = np.load(data_dir / "train_features_esm2.npy")
    X_test_pair = np.load(data_dir / "test_features_esm2.npy")
    
    # Load pairs to reconstruct individual embeddings (for Model 2B, 2C)
    with (data_dir / "train_pairs.pkl").open('rb') as f:
        train_pairs = pickle.load(f)
    with (data_dir / "test_pairs.pkl").open('rb') as f:
        test_pairs = pickle.load(f)
    
    # Load individual protein embeddings from cache
    esm_cache = load_feature_cache(ESM_CACHE_PATH)
    
    def get_embeddings(pairs):
        emb_a_list = []
        emb_b_list = []
        for id_a, id_b, seq_a, seq_b in pairs:
            emb_a = esm_cache.get(id_a, {}).get('embedding')
            emb_b = esm_cache.get(id_b, {}).get('embedding')
            if emb_a is None or emb_b is None:
                raise ValueError(f"Missing embeddings for pair ({id_a}, {id_b})")
            emb_a_list.append(emb_a)
            emb_b_list.append(emb_b)
        return np.array(emb_a_list), np.array(emb_b_list)
    
    train_emb_a, train_emb_b = get_embeddings(train_pairs)
    test_emb_a, test_emb_b = get_embeddings(test_pairs)
    
    return {
        'metadata': metadata,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_pair': X_train_pair,
        'X_test_pair': X_test_pair,
        'train_emb_a': train_emb_a,
        'train_emb_b': train_emb_b,
        'test_emb_a': test_emb_a,
        'test_emb_b': test_emb_b,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
    }


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int = 5,
    model_type: str = "2A",
    fold_idx: int = None,
) -> Dict:
    """Train the model with early stopping."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch in train_loader:
            if model_type in ["2B", "2C"]:
                batch_emb_a, batch_emb_b, batch_y = batch
                batch_emb_a = batch_emb_a.to(device)
                batch_emb_b = batch_emb_b.to(device)
                batch_y = batch_y.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(batch_emb_a, batch_emb_b)
            else:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = train_loss / train_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if model_type in ["2B", "2C"]:
                    batch_emb_a, batch_emb_b, batch_y = batch
                    batch_emb_a = batch_emb_a.to(device)
                    batch_emb_b = batch_emb_b.to(device)
                    batch_y = batch_y.to(device).float().unsqueeze(1)
                    outputs = model(batch_emb_a, batch_emb_b)
                elif model_type == "2D":
                    batch_seq_a, batch_seq_b, batch_y = batch
                    batch_seq_a = batch_seq_a.to(device)
                    batch_seq_b = batch_seq_b.to(device)
                    batch_y = batch_y.to(device).float().unsqueeze(1)
                    outputs = model(batch_seq_a, batch_seq_b)
                else:
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).float().unsqueeze(1)
                    outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            temp_checkpoint = f'best_model2{model_type}.pth'
            torch.save(model.state_dict(), temp_checkpoint)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            model.load_state_dict(torch.load(temp_checkpoint))
            break
    
    # Save final checkpoint
    if fold_idx is not None:
        checkpoint_path = CHECKPOINTS_DIR / f"fold_{fold_idx}_Model2{model_type}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': epoch + 1 - patience_counter,
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_type: str = "2A",
) -> Dict:
    """Evaluate the trained model on test set."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type in ["2B", "2C"]:
                batch_emb_a, batch_emb_b, batch_y = batch
                batch_emb_a = batch_emb_a.to(device)
                batch_emb_b = batch_emb_b.to(device)
                outputs = model(batch_emb_a, batch_emb_b)
            elif model_type == "2D":
                batch_seq_a, batch_seq_b, batch_y = batch
                batch_seq_a = batch_seq_a.to(device)
                batch_seq_b = batch_seq_b.to(device)
                outputs = model(batch_seq_a, batch_seq_b)
            else:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
            
            probs = outputs.cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
    }


def aggregate_metrics_across_folds(all_metrics_by_fold: List[List[Dict]]) -> Dict:
    """
    Aggregate metrics across folds, computing mean ± std.
    
    Args:
        all_metrics_by_fold: List of lists, where each inner list contains metrics for one fold
    
    Returns:
        Dictionary with aggregated metrics: {model_name: {metric: (mean, std)}}
    """
    # Structure: {model_name: {metric: [values]}}
    raw_data = {}
    
    for fold_metrics in all_metrics_by_fold:
        for m in fold_metrics:
            model_name = m.get('model_name', 'Unknown')
            
            if model_name not in raw_data:
                raw_data[model_name] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'roc_auc': [],
                    'pr_auc': [],
                }
            
            raw_data[model_name]['accuracy'].append(m['accuracy'])
            raw_data[model_name]['precision'].append(m['precision'])
            raw_data[model_name]['recall'].append(m['recall'])
            raw_data[model_name]['f1'].append(m['f1'])
            raw_data[model_name]['roc_auc'].append(m['roc_auc'])
            raw_data[model_name]['pr_auc'].append(m['pr_auc'])
    
    # Compute mean ± std
    aggregated = {}
    for model_name, metrics in raw_data.items():
        aggregated[model_name] = {}
        for metric, values in metrics.items():
            values_arr = np.array(values)
            aggregated[model_name][metric] = (
                np.mean(values_arr),
                np.std(values_arr)
            )
    
    return aggregated


def format_aggregated_metrics(aggregated: Dict) -> str:
    """Format aggregated metrics (mean ± std) as string for logging."""
    lines = []
    
    for model_name in sorted(aggregated.keys()):
        metrics = aggregated[model_name]
        lines.append(f"\n{model_name}:")
        lines.append(f"  Accuracy:   {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}")
        lines.append(f"  Precision:  {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}")
        lines.append(f"  Recall:     {metrics['recall'][0]:.4f} ± {metrics['recall'][1]:.4f}")
        lines.append(f"  F1-Score:   {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}")
        lines.append(f"  ROC-AUC:    {metrics['roc_auc'][0]:.4f} ± {metrics['roc_auc'][1]:.4f}")
        lines.append(f"  PR-AUC:     {metrics['pr_auc'][0]:.4f} ± {metrics['pr_auc'][1]:.4f}")
        lines.append("")
    
    return "\n".join(lines)


# Model 2D (D-SCRIPT) has been removed - use model2d_standalone.py instead


# ============================================================================
# PLOTTING AND LOGGING
# ============================================================================

def create_comparison_plot(all_metrics_by_fold: List[List[Dict]], output_path: Path):
    """
    Create bar plot comparing all Model 2 variants with error bars and individual fold values.
    
    Args:
        all_metrics_by_fold: List of lists, where each inner list contains metrics for one fold
        output_path: Path to save the plot
    """
    # Aggregate metrics across folds
    aggregated = {'Model 2A (Improved MLP)': [], 'Model 2B (Siamese MLP)': [], 'Model 2C (Transformer)': []}
    
    for fold_metrics in all_metrics_by_fold:
        for m in fold_metrics:
            model_name = m.get('model_name', 'Unknown')
            if model_name in aggregated:
                aggregated[model_name].append(m['roc_auc'])
    
    # Calculate means and stds
    model_names = []
    means = []
    stds = []
    all_values = []
    
    for model_name in ['Model 2A (Improved MLP)', 'Model 2B (Siamese MLP)', 'Model 2C (Transformer)']:
        if len(aggregated[model_name]) > 0:
            values = np.array(aggregated[model_name])
            model_names.append(model_name)
            means.append(np.mean(values))
            stds.append(np.std(values))
            all_values.append(values)
    
    if len(model_names) == 0:
        print("  No metrics to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(model_names))
    colors = ['#2e86ab', '#a23b72', '#f18f01']
    
    # Plot bars with error bars
    bars = ax.bar(x, means, yerr=stds, color=colors[:len(model_names)], alpha=0.8,
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add individual fold values as black dots
    for i, (model_name, values) in enumerate(zip(model_names, all_values)):
        x_positions = np.full(len(values), x[i])
        # Add small random jitter to avoid overlap
        jitter = np.random.normal(0, 0.05, len(values))
        ax.scatter(x_positions + jitter, values, color='black', s=30, alpha=0.6, zorder=5)
    
    # Add value labels on bars (mean ± std)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.02,
               f'{height:.3f}±{stds[i]:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    n_folds = len(all_metrics_by_fold)
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model 2 Variants - ROC-AUC Comparison ({n_folds}-Fold CV: Mean ± Std)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, max(1.05, max(means) + max(stds) + 0.1)])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def log_results(metadata: Dict, all_metrics_by_fold: List[List[Dict]], args, n_folds: int):
    """Write results to the results log file with aggregated metrics across folds."""
    with RESULTS_PATH.open('w') as fh:  # Changed from 'a' to 'w' to overwrite each run
        timestamp = datetime.utcnow().isoformat()
        fh.write("\n" + "="*70 + "\n")
        fh.write(f"Run timestamp (UTC): {timestamp}\n")
        fh.write(f"{n_folds}-Fold Cross-Validation Results\n")
        fh.write(f"Number of folds: {n_folds}\n")
        fh.write(f"Training parameters:\n")
        fh.write(f"  Learning rate: {args.learning_rate}\n")
        fh.write(f"  Batch size: {args.batch_size}\n")
        fh.write(f"  Epochs: {args.epochs}\n")
        fh.write(f"  Early stopping patience: {args.patience}\n")
        fh.write(f"  Device: {args.device}\n")
        fh.write(f"  Seed: {args.seed}\n")
        fh.write(f"\nDataset info:\n")
        
        # Use metadata from first fold
        if 'n_train' in metadata:
            fh.write(f"  Train samples: {metadata['n_train']} ({metadata.get('train_positive', 'N/A')} pos, {metadata.get('train_negative', 'N/A')} neg)\n")
            fh.write(f"  Test samples:  {metadata['n_test']} ({metadata.get('test_positive', 'N/A')} pos, {metadata.get('test_negative', 'N/A')} neg)\n")
        if 'seed' in metadata:
            fh.write(f"  Seed: {metadata['seed']}\n")
        fh.write("\n")
        
        # Aggregate metrics across folds
        aggregated = aggregate_metrics_across_folds(all_metrics_by_fold)
        
        fh.write("AGGREGATED RESULTS (Mean ± Std across folds):\n")
        fh.write(format_aggregated_metrics(aggregated))
        
        fh.write("\n" + "="*70 + "\n")


# ============================================================================
# SEED SETTING
# ============================================================================

def set_seed(seed: int = 18):
    """Set random seeds for reproducibility."""
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Create a generator for DataLoader reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train improved Model 2 variants for PPI prediction")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs (default: 20)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split (default: 0.1)")
    parser.add_argument("--fold", type=int, default=None, help="Train on specific fold only. If not specified, runs on all available folds.")
    parser.add_argument("--seed", type=int, default=18, help="Random seed for reproducibility (default: 18)")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    generator = set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Get available folds
    available_folds = get_available_folds()
    
    if not available_folds:
        print("="*80)
        print("MODEL 2: IMPROVED NEURAL NETWORK VARIANTS - K-FOLD CROSS-VALIDATION")
        print("="*80)
        print("\nERROR: No fold directories found in curated_data/")
        print("Please run prepare_dataset.py first to generate fold data.")
        return 1
    
    # Determine which folds to process
    if args.fold is not None:
        if args.fold not in available_folds:
            print(f"\nERROR: Fold {args.fold} not found. Available folds: {available_folds}")
            return 1
        folds_to_process = [args.fold]
    else:
        folds_to_process = available_folds
    
    print("="*80)
    print(f"MODEL 2: IMPROVED NEURAL NETWORK VARIANTS - {len(folds_to_process)}-FOLD CROSS-VALIDATION")
    print("="*80)
    
    if args.fold is not None:
        print(f"\nProcessing single fold: {args.fold}")
    else:
        print(f"\nProcessing {len(folds_to_process)} folds: {folds_to_process}")
    
    # Store metrics for each fold
    all_metrics_by_fold = []
    all_metadata = []
    
    # Process each fold
    for fold_idx in folds_to_process:
        print("\n" + "="*80)
        print(f"PROCESSING FOLD {fold_idx}")
        print("="*80)
        
        try:
            data = load_curated_data(fold_idx=fold_idx)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            continue
        
        metadata = data['metadata']
        all_metadata.append(metadata)
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"\nFold {fold_idx} info:")
        print(f"  Train samples: {len(y_train)} ({np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg)")
        print(f"  Test samples:  {len(y_test)} ({np.sum(y_test == 1)} pos, {np.sum(y_test == 0)} neg)")
        
        # Create validation split (using indices to keep pairs aligned)
        indices = np.arange(len(y_train))
        train_indices, val_indices = train_test_split(
            indices, test_size=args.val_split, random_state=args.seed, stratify=y_train
        )
        
        X_train_pair = data['X_train_pair'][train_indices]
        X_val_pair = data['X_train_pair'][val_indices]
        y_train_split = y_train[train_indices]
        y_val_split = y_train[val_indices]
        
        train_emb_a = data['train_emb_a'][train_indices]
        train_emb_b = data['train_emb_b'][train_indices]
        val_emb_a = data['train_emb_a'][val_indices]
        val_emb_b = data['train_emb_b'][val_indices]
        
        fold_metrics = []
        
        # Model 2A: Improved MLP
        print("\n" + "-"*80)
        print(f"FOLD {fold_idx} - MODEL 2A: Improved MLP")
        print("-"*80)
        
        X_train_tensor = torch.FloatTensor(X_train_pair)
        y_train_tensor = torch.LongTensor(y_train_split)
        X_val_tensor = torch.FloatTensor(X_val_pair)
        y_val_tensor = torch.LongTensor(y_val_split)
        X_test_tensor = torch.FloatTensor(data['X_test_pair'])
        y_test_tensor = torch.LongTensor(y_test)
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=args.batch_size, shuffle=False)
        
        model2a = Model2A_ImprovedMLP(input_dim=X_train_pair.shape[1])
        model2a.to(device)
        
        training_info = train_model(model2a, train_loader, val_loader, args.epochs, args.learning_rate, device, args.patience, "2A", fold_idx)
        metrics2a = evaluate_model(model2a, test_loader, device, "2A")
        fold_metrics.append({'model_name': 'Model 2A (Improved MLP)', **metrics2a})
        
        # Clean up
        if Path('best_model22A.pth').exists():
            Path('best_model22A.pth').unlink()
        
        # Model 2B: Siamese MLP
        print("\n" + "-"*80)
        print(f"FOLD {fold_idx} - MODEL 2B: Siamese MLP")
        print("-"*80)
        
        train_emb_a_tensor = torch.FloatTensor(train_emb_a)
        train_emb_b_tensor = torch.FloatTensor(train_emb_b)
        val_emb_a_tensor = torch.FloatTensor(val_emb_a)
        val_emb_b_tensor = torch.FloatTensor(val_emb_b)
        test_emb_a_tensor = torch.FloatTensor(data['test_emb_a'])
        test_emb_b_tensor = torch.FloatTensor(data['test_emb_b'])
        
        train_loader = DataLoader(TensorDataset(train_emb_a_tensor, train_emb_b_tensor, y_train_tensor), batch_size=args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(TensorDataset(val_emb_a_tensor, val_emb_b_tensor, y_val_tensor), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_emb_a_tensor, test_emb_b_tensor, y_test_tensor), batch_size=args.batch_size, shuffle=False)
        
        model2b = Model2B_SiameseMLP(protein_emb_dim=train_emb_a.shape[1])
        model2b.to(device)
        
        training_info = train_model(model2b, train_loader, val_loader, args.epochs, args.learning_rate, device, args.patience, "2B", fold_idx)
        metrics2b = evaluate_model(model2b, test_loader, device, "2B")
        fold_metrics.append({'model_name': 'Model 2B (Siamese MLP)', **metrics2b})
        
        if Path('best_model22B.pth').exists():
            Path('best_model22B.pth').unlink()
        
        # Model 2C: Transformer Classifier
        print("\n" + "-"*80)
        print(f"FOLD {fold_idx} - MODEL 2C: Transformer Classifier")
        print("-"*80)
        
        model2c = Model2C_TransformerClassifier(protein_emb_dim=train_emb_a.shape[1])
        model2c.to(device)
        
        training_info = train_model(model2c, train_loader, val_loader, args.epochs, args.learning_rate, device, args.patience, "2C", fold_idx)
        metrics2c = evaluate_model(model2c, test_loader, device, "2C")
        fold_metrics.append({'model_name': 'Model 2C (Transformer)', **metrics2c})
        
        if Path('best_model22C.pth').exists():
            Path('best_model22C.pth').unlink()
        
        all_metrics_by_fold.append(fold_metrics)
        print(f"\n✓ Fold {fold_idx} complete ({len(fold_metrics)} models evaluated)")
    
    # Aggregate results across folds
    if all_metrics_by_fold:
        print("\n" + "="*80)
        print("AGGREGATED RESULTS ACROSS ALL FOLDS")
        print("="*80)
        
        aggregated = aggregate_metrics_across_folds(all_metrics_by_fold)
        
        # Print summary
        for model_name in ['Model 2A (Improved MLP)', 'Model 2B (Siamese MLP)', 'Model 2C (Transformer)']:
            if model_name in aggregated:
                metrics = aggregated[model_name]
                print(f"\n{model_name}:")
                print(f"  ROC-AUC: {metrics['roc_auc'][0]:.4f} ± {metrics['roc_auc'][1]:.4f}")
                print(f"  F1-Score: {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}")
        
        # Create plot
        if len(all_metrics_by_fold) > 1:  # Only plot if multiple folds
            plot_path = PLOT_DIR / "model2_comparison.png"
            create_comparison_plot(all_metrics_by_fold, plot_path)
        
        # Log results
        if all_metadata:
            log_results(all_metadata[0], all_metrics_by_fold, args, len(folds_to_process))
            print(f"\nResults saved to: {RESULTS_PATH}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
