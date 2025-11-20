#!/usr/bin/env python3
"""
Benchmark script for Protein-Protein Interaction (PPI) prediction.

This script loads pre-computed features from curated_data/ and trains multiple classifiers:
- Model 1A: ESM-2 embedding + various classifiers
- Model 1B: Classical handcrafted-feature embedding + various classifiers

Classifiers tested:
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- K-Nearest Neighbors (KNN)

Run prepare_dataset.py first to generate the curated datasets.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# Try to import optional tree-based models
# Use lazy imports to handle library loading errors gracefully
LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False

def _check_lightgbm():
    """Check if LightGBM is available and can be imported."""
    global LIGHTGBM_AVAILABLE
    if LIGHTGBM_AVAILABLE:
        return True
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
        return True
    except (ImportError, OSError, Exception):
        return False

def _check_xgboost():
    """Check if XGBoost is available and can be imported."""
    global XGBOOST_AVAILABLE
    if XGBOOST_AVAILABLE:
        return True
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
        return True
    except (ImportError, OSError, Exception):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
RESULTS_PATH = PROJECT_ROOT / "model1_results.txt"
PLOT_DIR = PROJECT_ROOT / "plot"
PLOT_DIR.mkdir(exist_ok=True)


def load_curated_data(fold_idx: int = None):
    """
    Load pre-computed features and labels from curated_data/.
    
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
    
    # Load labels
    y_train = np.load(data_dir / "train_labels.npy")
    y_test = np.load(data_dir / "test_labels.npy")
    
    # Load handcrafted features
    X_train_handcraft = np.load(data_dir / "train_features_handcrafted.npy")
    X_test_handcraft = np.load(data_dir / "test_features_handcrafted.npy")
    
    # Load ESM-2 features (if available)
    X_train_esm = None
    X_test_esm = None
    if metadata.get("esm2_available", False):
        esm_train_path = data_dir / "train_features_esm2.npy"
        esm_test_path = data_dir / "test_features_esm2.npy"
        if esm_train_path.exists() and esm_test_path.exists():
            X_train_esm = np.load(esm_train_path)
            X_test_esm = np.load(esm_test_path)
    
    return {
        'metadata': metadata,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_handcraft': X_train_handcraft,
        'X_test_handcraft': X_test_handcraft,
        'X_train_esm': X_train_esm,
        'X_test_esm': X_test_esm,
    }


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


def get_classifier(name: str):
    """Get classifier instance by name."""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    elif name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == "LightGBM":
        if not _check_lightgbm():
            raise ImportError(
                "LightGBM not available. Install with: pip install lightgbm\n"
                "Note: On macOS, you may need: conda install -c conda-forge lightgbm"
            )
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    elif name == "XGBoost":
        if not _check_xgboost():
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier_name: str,
) -> Dict:
    """
    Train a classifier and evaluate on test set.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        classifier_name: Name of the classifier
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nTraining {classifier_name}...")
    
    try:
        clf = get_classifier(classifier_name)
    except (ImportError, ValueError) as e:
        print(f"  ERROR: {e}")
        return None
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "classifier": classifier_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def train_all_classifiers(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_type: str,
) -> List[Dict]:
    """
    Train all available classifiers and return metrics.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        feature_type: "ESM-2" or "Handcrafted"
        
    Returns:
        List of metric dictionaries
    """
    classifiers = ["LogisticRegression", "RandomForest", "KNN"]
    
    if _check_lightgbm():
        classifiers.append("LightGBM")
    if _check_xgboost():
        classifiers.append("XGBoost")
    
    all_metrics = []
    
    for clf_name in classifiers:
        metrics = train_and_evaluate(X_train, X_test, y_train, y_test, clf_name)
        if metrics:
            metrics['feature_type'] = feature_type
            all_metrics.append(metrics)
    
    return all_metrics


def print_metrics(metrics: Dict, model_name: str):
    """Print formatted evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{model_name} - {metrics['classifier']} - Evaluation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  [[{cm[0, 0]:4d} {cm[0, 1]:4d}]")
    print(f"   [{cm[1, 0]:4d} {cm[1, 1]:4d}]]")


def format_metrics(metrics: Dict | None) -> str:
    """Format metrics dictionary as string for logging."""
    if metrics is None:
        return "  Not available\n"
    
    lines = []
    lines.append(f"  Classifier: {metrics['classifier']}")
    lines.append(f"  Accuracy:   {metrics['accuracy']:.4f}")
    lines.append(f"  Precision:  {metrics['precision']:.4f}")
    lines.append(f"  Recall:     {metrics['recall']:.4f}")
    lines.append(f"  F1-Score:   {metrics['f1']:.4f}")
    lines.append(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    lines.append(f"  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    lines.append(f"    [[{cm[0, 0]:4d} {cm[0, 1]:4d}]")
    lines.append(f"     [{cm[1, 0]:4d} {cm[1, 1]:4d}]]")
    return "\n".join(lines) + "\n"


def aggregate_metrics_across_folds(all_metrics_by_fold: List[List[Dict]]) -> Dict:
    """
    Aggregate metrics across folds, computing mean ± std.
    
    Args:
        all_metrics_by_fold: List of lists, where each inner list contains metrics for one fold
    
    Returns:
        Dictionary with aggregated metrics: {feature_type: {classifier: {metric: (mean, std)}}}
    """
    # Structure: {feature_type: {classifier: {metric: [values]}}}
    raw_data = {}
    
    for fold_metrics in all_metrics_by_fold:
        for m in fold_metrics:
            feature_type = m.get('feature_type', 'Unknown')
            classifier = m['classifier']
            
            if feature_type not in raw_data:
                raw_data[feature_type] = {}
            if classifier not in raw_data[feature_type]:
                raw_data[feature_type][classifier] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'roc_auc': [],
                }
            
            raw_data[feature_type][classifier]['accuracy'].append(m['accuracy'])
            raw_data[feature_type][classifier]['precision'].append(m['precision'])
            raw_data[feature_type][classifier]['recall'].append(m['recall'])
            raw_data[feature_type][classifier]['f1'].append(m['f1'])
            raw_data[feature_type][classifier]['roc_auc'].append(m['roc_auc'])
    
    # Compute mean ± std
    aggregated = {}
    for feature_type, classifiers in raw_data.items():
        aggregated[feature_type] = {}
        for classifier, metrics in classifiers.items():
            aggregated[feature_type][classifier] = {}
            for metric, values in metrics.items():
                values_arr = np.array(values)
                aggregated[feature_type][classifier][metric] = (
                    np.mean(values_arr),
                    np.std(values_arr)
                )
    
    return aggregated


def format_aggregated_metrics(aggregated: Dict) -> str:
    """Format aggregated metrics (mean ± std) as string for logging."""
    lines = []
    
    for feature_type in ['ESM-2', 'Handcrafted']:
        if feature_type not in aggregated:
            continue
        
        lines.append(f"\n{feature_type} Features:")
        for classifier in sorted(aggregated[feature_type].keys()):
            metrics = aggregated[feature_type][classifier]
            lines.append(f"  {classifier}:")
            lines.append(f"    Accuracy:   {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}")
            lines.append(f"    Precision:  {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}")
            lines.append(f"    Recall:     {metrics['recall'][0]:.4f} ± {metrics['recall'][1]:.4f}")
            lines.append(f"    F1-Score:   {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}")
            lines.append(f"    ROC-AUC:    {metrics['roc_auc'][0]:.4f} ± {metrics['roc_auc'][1]:.4f}")
            lines.append("")
    
    return "\n".join(lines)


def create_roc_auc_plot(all_metrics_by_fold: List[List[Dict]], output_path: Path):
    """
    Create a bar plot of ROC-AUC scores with error bars and individual fold values.
    
    Args:
        all_metrics_by_fold: List of lists, where each inner list contains metrics for one fold
        output_path: Path to save the plot
    """
    # Aggregate metrics across folds
    # Structure: {feature_type: {classifier: [roc_auc_values]}}
    aggregated = {'ESM-2': {}, 'Handcrafted': {}}
    
    for fold_metrics in all_metrics_by_fold:
        for m in fold_metrics:
            feature_type = m.get('feature_type', 'Unknown')
            classifier = m['classifier']
            roc_auc = m['roc_auc']
            
            if feature_type in aggregated:
                if classifier not in aggregated[feature_type]:
                    aggregated[feature_type][classifier] = []
                aggregated[feature_type][classifier].append(roc_auc)
    
    # Get all unique classifiers
    all_classifiers = sorted(set(
        list(aggregated['ESM-2'].keys()) + list(aggregated['Handcrafted'].keys())
    ))
    
    # Calculate means and stds
    esm_means = []
    esm_stds = []
    esm_all_values = []  # For individual points
    
    handcraft_means = []
    handcraft_stds = []
    handcraft_all_values = []  # For individual points
    
    for clf in all_classifiers:
        # ESM-2
        if clf in aggregated['ESM-2']:
            values = np.array(aggregated['ESM-2'][clf])
            esm_means.append(np.mean(values))
            esm_stds.append(np.std(values))
            esm_all_values.append(values)
        else:
            esm_means.append(0)
            esm_stds.append(0)
            esm_all_values.append([])
        
        # Handcrafted
        if clf in aggregated['Handcrafted']:
            values = np.array(aggregated['Handcrafted'][clf])
            handcraft_means.append(np.mean(values))
            handcraft_stds.append(np.std(values))
            handcraft_all_values.append(values)
        else:
            handcraft_means.append(0)
            handcraft_stds.append(0)
            handcraft_all_values.append([])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(all_classifiers))
    width = 0.35
    
    # Plot bars with error bars
    bars1 = ax.bar(x - width/2, esm_means, width, yerr=esm_stds, 
                   label='Model 1A (ESM-2)', color='#2e86ab', alpha=0.8, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2 = ax.bar(x + width/2, handcraft_means, width, yerr=handcraft_stds,
                   label='Model 1B (Handcrafted)', color='#a23b72', alpha=0.8,
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add individual fold values as black dots
    for i, clf in enumerate(all_classifiers):
        # ESM-2 individual points
        if len(esm_all_values[i]) > 0:
            x_positions = np.full(len(esm_all_values[i]), x[i] - width/2)
            # Add small random jitter to avoid overlap
            jitter = np.random.normal(0, width/8, len(esm_all_values[i]))
            ax.scatter(x_positions + jitter, esm_all_values[i], 
                      color='black', s=30, alpha=0.6, zorder=5)
        
        # Handcrafted individual points
        if len(handcraft_all_values[i]) > 0:
            x_positions = np.full(len(handcraft_all_values[i]), x[i] + width/2)
            # Add small random jitter to avoid overlap
            jitter = np.random.normal(0, width/8, len(handcraft_all_values[i]))
            ax.scatter(x_positions + jitter, handcraft_all_values[i],
                      color='black', s=30, alpha=0.6, zorder=5)
    
    # Add value labels on bars (mean ± std)
    def autolabel(bars, means, stds):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.02,
                       f'{height:.3f}±{stds[i]:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    autolabel(bars1, esm_means, esm_stds)
    autolabel(bars2, handcraft_means, handcraft_stds)
    
    # Customize plot
    n_folds = len(all_metrics_by_fold)
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC-AUC Scores by Classifier and Feature Type ({n_folds}-Fold CV: Mean ± Std)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classifiers, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(1.05, max(esm_means + handcraft_means) + max(esm_stds + handcraft_stds) + 0.1)])
    
    # Add horizontal line at 0.5 (random classifier)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nROC-AUC plot with error bars saved to: {output_path}")
    plt.close()


def log_model_results(metadata: Dict, all_metrics_by_fold: List[List[Dict]], n_folds: int):
    """Write results to the results log file with aggregated metrics across folds."""
    with RESULTS_PATH.open('w') as fh:  # Changed from 'a' to 'w' to overwrite each run
        timestamp = datetime.utcnow().isoformat()
        fh.write("\n" + "="*70 + "\n")
        fh.write(f"Run timestamp (UTC): {timestamp}\n")
        fh.write(f"{n_folds}-Fold Cross-Validation Results\n")
        fh.write(f"Number of folds: {n_folds}\n")
        fh.write(f"Dataset info:\n")
        
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


def main():
    parser = argparse.ArgumentParser(description="PPI prediction benchmark using curated data with K-fold CV")
    parser.add_argument(
        "--skip-esm",
        action="store_true",
        help="Skip Model 1A (ESM-2 embeddings)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Train only specific classifier (LogisticRegression, RandomForest, LightGBM, XGBoost, KNN)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train on specific fold only. If not specified, runs on all available folds.",
    )
    args = parser.parse_args()
    
    # Get available folds
    available_folds = get_available_folds()
    
    if not available_folds:
        print("="*80)
        print("PPI PREDICTION BENCHMARK - K-FOLD CROSS-VALIDATION")
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
    print(f"PPI PREDICTION BENCHMARK - {len(folds_to_process)}-FOLD CROSS-VALIDATION")
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
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue
        
        metadata = data['metadata']
        all_metadata.append(metadata)
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"\nFold {fold_idx} info:")
        print(f"  Train samples: {len(y_train)} ({np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg)")
        print(f"  Test samples:  {len(y_test)} ({np.sum(y_test == 1)} pos, {np.sum(y_test == 0)} neg)")
        
        fold_metrics = []
        
        # Model 1A: ESM-2 embeddings
        if not args.skip_esm:
            if data['X_train_esm'] is not None:
                X_train_esm = data['X_train_esm']
                X_test_esm = data['X_test_esm']
                
                if args.classifier:
                    metrics = train_and_evaluate(X_train_esm, X_test_esm, y_train, y_test, args.classifier)
                    if metrics:
                        metrics['feature_type'] = 'ESM-2'
                        fold_metrics.append(metrics)
                else:
                    esm_metrics = train_all_classifiers(X_train_esm, X_test_esm, y_train, y_test, "ESM-2")
                    fold_metrics.extend(esm_metrics)
        
        # Model 1B: Handcrafted features
        X_train_handcraft = data['X_train_handcraft']
        X_test_handcraft = data['X_test_handcraft']
        
        if args.classifier:
            metrics = train_and_evaluate(X_train_handcraft, X_test_handcraft, y_train, y_test, args.classifier)
            if metrics:
                metrics['feature_type'] = 'Handcrafted'
                fold_metrics.append(metrics)
        else:
            handcraft_metrics = train_all_classifiers(X_train_handcraft, X_test_handcraft, y_train, y_test, "Handcrafted")
            fold_metrics.extend(handcraft_metrics)
        
        all_metrics_by_fold.append(fold_metrics)
        print(f"\n✓ Fold {fold_idx} complete ({len(fold_metrics)} models evaluated)")
    
    # Aggregate results across folds
    if all_metrics_by_fold:
        print("\n" + "="*80)
        print("AGGREGATED RESULTS ACROSS ALL FOLDS")
        print("="*80)
        
        aggregated = aggregate_metrics_across_folds(all_metrics_by_fold)
        
        # Print summary
        for feature_type in ['ESM-2', 'Handcrafted']:
            if feature_type not in aggregated:
                continue
            
            print(f"\n{feature_type} Features:")
            print("-" * 80)
            for classifier in sorted(aggregated[feature_type].keys()):
                metrics = aggregated[feature_type][classifier]
                print(f"  {classifier:20s} - ROC-AUC: {metrics['roc_auc'][0]:.4f} ± {metrics['roc_auc'][1]:.4f}")
        
        # Create ROC-AUC plot with error bars
        if len(all_metrics_by_fold) > 1:  # Only plot if multiple folds
            plot_path = PLOT_DIR / "roc_auc_comparison.png"
            create_roc_auc_plot(all_metrics_by_fold, plot_path)
        
        # Log results
        if all_metadata:
            log_model_results(all_metadata[0], all_metrics_by_fold, len(folds_to_process))
            print(f"\nResults written to: {RESULTS_PATH}")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
