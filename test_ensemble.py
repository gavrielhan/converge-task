#!/usr/bin/env python3
"""
Test Ensemble Method: LightGBM + Model2B Fallback

Tests the hybrid prediction strategy where Model2B is used when LightGBM
confidence is below 65%. Compares performance against using LightGBM alone.
"""

import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ESM_CACHE_PATH = PROJECT_ROOT / "cache" / "esm_embeddings.pkl"


# Model2B architecture (must match training)
class Model2B_SiameseMLP(nn.Module):
    """Model 2B: Siamese MLP architecture."""
    
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


def load_fold_data(fold_idx: int):
    """Load test data for a specific fold."""
    fold_dir = CURATED_DATA_DIR / f"fold_{fold_idx}"
    
    # Load test data
    X_test_esm = np.load(fold_dir / "test_features_esm2.npy")
    y_test = np.load(fold_dir / "test_labels.npy")
    
    # Load test pairs for Model 2B (need individual embeddings)
    with (fold_dir / "test_pairs.pkl").open('rb') as f:
        test_pairs = pickle.load(f)
    
    # Load individual protein embeddings from cache
    if ESM_CACHE_PATH.exists():
        with ESM_CACHE_PATH.open('rb') as f:
            esm_cache = pickle.load(f)
        
        test_emb_a_list = []
        test_emb_b_list = []
        for id_a, id_b, seq_a, seq_b in test_pairs:
            emb_a = esm_cache.get(id_a, {}).get('embedding')
            emb_b = esm_cache.get(id_b, {}).get('embedding')
            if emb_a is not None and emb_b is not None:
                test_emb_a_list.append(emb_a)
                test_emb_b_list.append(emb_b)
        
        test_emb_a = np.array(test_emb_a_list)
        test_emb_b = np.array(test_emb_b_list)
    else:
        test_emb_a = None
        test_emb_b = None
    
    return X_test_esm, y_test, test_emb_a, test_emb_b


def load_models(fold_idx: int):
    """Load LightGBM and Model2B for a specific fold."""
    # Load LightGBM
    lgbm_path = CHECKPOINTS_DIR / "model1" / f"fold_{fold_idx}_esm2_LightGBM.pkl"
    if not lgbm_path.exists():
        raise FileNotFoundError(f"LightGBM checkpoint not found: {lgbm_path}")
    lgbm_model = joblib.load(lgbm_path)
    
    # Load Model2B
    model2b_path = CHECKPOINTS_DIR / "model2" / f"fold_{fold_idx}_Model22B.pth"
    if not model2b_path.exists():
        raise FileNotFoundError(f"Model2B checkpoint not found: {model2b_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2b = Model2B_SiameseMLP(protein_emb_dim=1280)
    model2b.load_state_dict(torch.load(model2b_path, map_location=device))
    model2b = model2b.to(device)
    model2b.eval()
    
    return lgbm_model, model2b, device


def predict_ensemble(lgbm_model, model2b, X_test_esm, test_emb_a, test_emb_b, 
                     device, confidence_threshold=0.65):
    """
    Predict using ensemble method:
    - Use LightGBM if confidence >= threshold
    - Use Model2B if confidence < threshold
    
    Returns:
        predictions, probabilities, model_used_flags
    """
    # Get LightGBM predictions
    lgbm_proba = lgbm_model.predict_proba(X_test_esm)
    lgbm_pred = lgbm_model.predict(X_test_esm)
    
    # Calculate confidence for each prediction
    lgbm_confidence = np.max(lgbm_proba, axis=1)
    
    # Identify low confidence samples
    low_confidence_mask = lgbm_confidence < confidence_threshold
    n_low_confidence = low_confidence_mask.sum()
    
    print(f"    Low confidence samples: {n_low_confidence}/{len(lgbm_pred)} ({100*n_low_confidence/len(lgbm_pred):.1f}%)")
    
    # Initialize with LightGBM predictions
    ensemble_pred = lgbm_pred.copy()
    ensemble_proba = lgbm_proba.copy()
    model_used = np.array(['LightGBM'] * len(lgbm_pred))
    
    # Use Model2B for low confidence samples
    if n_low_confidence > 0 and test_emb_a is not None:
        # Get indices of low confidence samples
        low_conf_indices = np.where(low_confidence_mask)[0]
        
        # Prepare data for Model2B
        emb_a_low = torch.FloatTensor(test_emb_a[low_conf_indices]).to(device)
        emb_b_low = torch.FloatTensor(test_emb_b[low_conf_indices]).to(device)
        
        # Predict with Model2B
        with torch.no_grad():
            model2b_output = model2b(emb_a_low, emb_b_low)
            model2b_prob_interact = model2b_output.cpu().numpy().flatten()
        
        # Update predictions for low confidence samples
        model2b_proba = np.column_stack([1 - model2b_prob_interact, model2b_prob_interact])
        model2b_pred = (model2b_prob_interact > 0.5).astype(int)
        
        ensemble_pred[low_conf_indices] = model2b_pred
        ensemble_proba[low_conf_indices] = model2b_proba
        model_used[low_conf_indices] = 'Model2B'
    
    return ensemble_pred, ensemble_proba, model_used


def evaluate_predictions(y_true, y_pred, y_proba):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def test_ensemble():
    """Test ensemble method across all folds."""
    print("="*80)
    print("TESTING ENSEMBLE METHOD: LightGBM + Model2B Fallback")
    print("="*80)
    
    # Determine number of folds
    n_folds = len(list(CURATED_DATA_DIR.glob("fold_*")))
    print(f"\n✓ Found {n_folds} folds")
    
    # Test different confidence thresholds
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    
    results_by_threshold = {thresh: [] for thresh in thresholds}
    lgbm_only_results = []
    model2b_only_results = []
    
    for fold_idx in range(n_folds):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold_idx}")
        print(f"{'─'*80}")
        
        # Load data
        X_test_esm, y_test, test_emb_a, test_emb_b = load_fold_data(fold_idx)
        print(f"  Test samples: {len(y_test)}")
        
        # Load models
        try:
            lgbm_model, model2b, device = load_models(fold_idx)
            print(f"  ✓ Models loaded")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue
        
        # Test LightGBM only
        lgbm_proba = lgbm_model.predict_proba(X_test_esm)
        lgbm_pred = lgbm_model.predict(X_test_esm)
        lgbm_metrics = evaluate_predictions(y_test, lgbm_pred, lgbm_proba)
        lgbm_only_results.append(lgbm_metrics)
        print(f"\n  LightGBM only:")
        print(f"    ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")
        print(f"    Accuracy: {lgbm_metrics['accuracy']:.4f}")
        
        # Test Model2B only
        if test_emb_a is not None:
            emb_a_tensor = torch.FloatTensor(test_emb_a).to(device)
            emb_b_tensor = torch.FloatTensor(test_emb_b).to(device)
            
            with torch.no_grad():
                model2b_output = model2b(emb_a_tensor, emb_b_tensor)
                model2b_prob_interact = model2b_output.cpu().numpy().flatten()
            
            model2b_proba = np.column_stack([1 - model2b_prob_interact, model2b_prob_interact])
            model2b_pred = (model2b_prob_interact > 0.5).astype(int)
            model2b_metrics = evaluate_predictions(y_test, model2b_pred, model2b_proba)
            model2b_only_results.append(model2b_metrics)
            print(f"\n  Model2B only:")
            print(f"    ROC-AUC: {model2b_metrics['roc_auc']:.4f}")
            print(f"    Accuracy: {model2b_metrics['accuracy']:.4f}")
        
        # Test ensemble with different thresholds
        print(f"\n  Ensemble (LightGBM + Model2B fallback):")
        for threshold in thresholds:
            ensemble_pred, ensemble_proba, model_used = predict_ensemble(
                lgbm_model, model2b, X_test_esm, test_emb_a, test_emb_b, 
                device, confidence_threshold=threshold
            )
            
            ensemble_metrics = evaluate_predictions(y_test, ensemble_pred, ensemble_proba)
            results_by_threshold[threshold].append(ensemble_metrics)
            
            n_model2b = (model_used == 'Model2B').sum()
            print(f"    Threshold {threshold:.2f}: ROC-AUC {ensemble_metrics['roc_auc']:.4f} "
                  f"(Model2B used: {n_model2b}/{len(y_test)})")
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    # LightGBM only
    if lgbm_only_results:
        lgbm_roc_aucs = [r['roc_auc'] for r in lgbm_only_results]
        print(f"\nLightGBM only:")
        print(f"  ROC-AUC: {np.mean(lgbm_roc_aucs):.4f} ± {np.std(lgbm_roc_aucs):.4f}")
    
    # Model2B only
    if model2b_only_results:
        model2b_roc_aucs = [r['roc_auc'] for r in model2b_only_results]
        print(f"\nModel2B only:")
        print(f"  ROC-AUC: {np.mean(model2b_roc_aucs):.4f} ± {np.std(model2b_roc_aucs):.4f}")
    
    # Ensemble with different thresholds
    print(f"\nEnsemble (LightGBM + Model2B fallback):")
    best_threshold = None
    best_roc_auc = 0
    
    for threshold in thresholds:
        if results_by_threshold[threshold]:
            roc_aucs = [r['roc_auc'] for r in results_by_threshold[threshold]]
            mean_roc_auc = np.mean(roc_aucs)
            std_roc_auc = np.std(roc_aucs)
            
            improvement = mean_roc_auc - np.mean(lgbm_roc_aucs)
            
            print(f"  Threshold {threshold:.2f}: ROC-AUC {mean_roc_auc:.4f} ± {std_roc_auc:.4f} "
                  f"(Δ {improvement:+.4f})")
            
            if mean_roc_auc > best_roc_auc:
                best_roc_auc = mean_roc_auc
                best_threshold = threshold
    
    print(f"\n{'='*80}")
    print(f"BEST ENSEMBLE CONFIGURATION")
    print(f"{'='*80}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"ROC-AUC: {best_roc_auc:.4f}")
    print(f"Improvement over LightGBM only: {best_roc_auc - np.mean(lgbm_roc_aucs):+.4f}")
    
    if best_roc_auc > np.mean(lgbm_roc_aucs):
        print("\n✓ Ensemble method improves performance!")
    else:
        print("\n✗ Ensemble method does not improve performance.")
        print("  LightGBM alone is sufficient for this task.")


if __name__ == "__main__":
    test_ensemble()

