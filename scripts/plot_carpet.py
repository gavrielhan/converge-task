#!/usr/bin/env python3
"""
Carpet Plot: Visualize model predictions across folds to identify error patterns.

This plot shows:
- Y-axis: Test samples grouped by fold
- X-axis: Different models (Classical ML + Neural Networks)
- Colors: Green = Correct prediction, Red = Incorrect prediction

The goal is to see:
1. Which samples are consistently hard across all models
2. Whether different models make complementary errors (ensemble potential)
3. Performance patterns across different folds
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import joblib

warnings.filterwarnings("ignore")

# Import PyTorch for neural models
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
PLOT_DIR = PROJECT_ROOT / "plot"
PLOT_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Model 2 architectures (must match model.py)
class Model2A_ImprovedMLP(nn.Module):
    """Model 2A: Better MLP with GELU, LayerNorm, wider layers."""
    
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


class Model2C_TransformerClassifier(nn.Module):
    """Model 2C: Transformer encoder on ESM-2 embeddings."""
    
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


def load_fold_data(fold_idx: int):
    """Load train and test data for a specific fold."""
    fold_dir = CURATED_DATA_DIR / f"fold_{fold_idx}"
    
    # Load test data
    test_data = {
        'X_esm': np.load(fold_dir / "test_features_esm2.npy"),
        'y': np.load(fold_dir / "test_labels.npy")
    }
    
    # Load test pairs for Model 2B/2C (need individual embeddings)
    with (fold_dir / "test_pairs.pkl").open('rb') as f:
        test_pairs = pickle.load(f)
    
    # Load individual protein embeddings from cache
    esm_cache_path = PROJECT_ROOT / "cache" / "esm_embeddings.pkl"
    if esm_cache_path.exists():
        with esm_cache_path.open('rb') as f:
            esm_cache = pickle.load(f)
        
        test_emb_a_list = []
        test_emb_b_list = []
        for id_a, id_b, seq_a, seq_b in test_pairs:
            emb_a = esm_cache.get(id_a, {}).get('embedding')
            emb_b = esm_cache.get(id_b, {}).get('embedding')
            if emb_a is not None and emb_b is not None:
                test_emb_a_list.append(emb_a)
                test_emb_b_list.append(emb_b)
        
        test_data['test_emb_a'] = np.array(test_emb_a_list)
        test_data['test_emb_b'] = np.array(test_emb_b_list)
    
    return test_data


def predict_classical(checkpoint_path: Path, X_test: np.ndarray) -> np.ndarray:
    """Load a classical ML model checkpoint and predict."""
    try:
        model = joblib.load(checkpoint_path)
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        print(f"    ✗ Error loading {checkpoint_path.name}: {e}")
        return None


def predict_neural(checkpoint_path: Path, X_test: np.ndarray, model_type: str, 
                   test_emb_a: np.ndarray = None, test_emb_b: np.ndarray = None) -> np.ndarray:
    """Load a neural network checkpoint and predict.
    
    Args:
        model_type: "A", "B", or "C" for Model2A, Model2B, Model2C
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        if model_type == "A":
            model = Model2A_ImprovedMLP(input_dim=X_test.shape[1])
            X_test_t = torch.FloatTensor(X_test).to(device)
        elif model_type == "B":
            if test_emb_a is None or test_emb_b is None:
                return None
            model = Model2B_SiameseMLP(protein_emb_dim=test_emb_a.shape[1])
            test_emb_a_t = torch.FloatTensor(test_emb_a).to(device)
            test_emb_b_t = torch.FloatTensor(test_emb_b).to(device)
        elif model_type == "C":
            if test_emb_a is None or test_emb_b is None:
                return None
            model = Model2C_TransformerClassifier(protein_emb_dim=test_emb_a.shape[1])
            test_emb_a_t = torch.FloatTensor(test_emb_a).to(device)
            test_emb_b_t = torch.FloatTensor(test_emb_b).to(device)
        else:
            return None
        
        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Predict
        with torch.no_grad():
            if model_type == "A":
                outputs = model(X_test_t)
            else:
                outputs = model(test_emb_a_t, test_emb_b_t)
            
            probs = outputs.cpu().numpy().flatten()
            y_pred = (probs >= 0.5).astype(int)
        
        return y_pred
    except Exception as e:
        print(f"    ✗ Error loading {checkpoint_path.name}: {e}")
        return None


def create_carpet_plot():
    """Create carpet plot showing prediction correctness across folds and models."""
    
    print("="*80)
    print("CREATING CARPET PLOT FROM SAVED CHECKPOINTS")
    print("="*80)
    
    # Check if checkpoints exist
    if not CHECKPOINTS_DIR.exists():
        print(f"\n✗ ERROR: Checkpoints directory not found: {CHECKPOINTS_DIR}")
        print("Please run benchmark.py and model.py first to train models and save checkpoints.")
        return
    
    # Determine number of folds
    n_folds = len(list(CURATED_DATA_DIR.glob("fold_*")))
    print(f"\n✓ Found {n_folds} folds")
    
    # Define models to evaluate (LightGBM first for ordering)
    classical_models = [
        ("LightGBM", "LightGBM"),
        ("XGBoost", "XGBoost"),
        ("RandomForest", "Random Forest"),
        ("LogisticRegression", "Logistic Regression"),
        ("KNN", "KNN"),
    ]
    
    neural_models = [
        ("A", "Classic MLP"),
        ("B", "Siamese MLP"),
        ("C", "Transformer"),
    ]
    
    # Collect predictions for each fold and model
    all_predictions = {}
    all_labels = []
    fold_boundaries = [0]  # Track where each fold starts
    model_display_names = []
    
    for fold_idx in range(n_folds):
        print(f"\n{'─'*80}")
        print(f"Processing Fold {fold_idx}...")
        print(f"{'─'*80}")
        
        test_data = load_fold_data(fold_idx)
        X_test_esm = test_data['X_esm']
        y_test = test_data['y']
        test_emb_a = test_data.get('test_emb_a')
        test_emb_b = test_data.get('test_emb_b')
        
        all_labels.extend(y_test)
        fold_boundaries.append(fold_boundaries[-1] + len(y_test))
        
        # Classical models on ESM-2 embeddings
        for model_key, model_display_name in classical_models:
            checkpoint_path = CHECKPOINTS_DIR / "model1" / f"fold_{fold_idx}_esm2_{model_key}.pkl"
            
            if checkpoint_path.exists():
                print(f"  Loading {model_display_name}...", end=" ")
                y_pred = predict_classical(checkpoint_path, X_test_esm)
                
                if y_pred is not None:
                    if model_display_name not in all_predictions:
                        all_predictions[model_display_name] = []
                        model_display_names.append(model_display_name)
                    all_predictions[model_display_name].extend(y_pred)
                    acc = np.mean(y_pred == y_test)
                    print(f"✓ Acc: {acc:.3f}")
                else:
                    if model_display_name not in all_predictions:
                        all_predictions[model_display_name] = []
                        model_display_names.append(model_display_name)
                    all_predictions[model_display_name].extend([0] * len(y_test))
            else:
                print(f"  ⚠️  {model_display_name} checkpoint not found")
                if model_display_name not in all_predictions:
                    all_predictions[model_display_name] = []
                    model_display_names.append(model_display_name)
                all_predictions[model_display_name].extend([0] * len(y_test))
        
        # Neural models on ESM-2 embeddings
        for model_key, model_display_name in neural_models:
            checkpoint_path = CHECKPOINTS_DIR / "model2" / f"fold_{fold_idx}_Model22{model_key}.pth"
            
            if checkpoint_path.exists():
                print(f"  Loading {model_display_name}...", end=" ")
                y_pred = predict_neural(checkpoint_path, X_test_esm, model_key, test_emb_a, test_emb_b)
                
                if y_pred is not None:
                    if model_display_name not in all_predictions:
                        all_predictions[model_display_name] = []
                        model_display_names.append(model_display_name)
                    all_predictions[model_display_name].extend(y_pred)
                    acc = np.mean(y_pred == y_test)
                    print(f"✓ Acc: {acc:.3f}")
                else:
                    if model_display_name not in all_predictions:
                        all_predictions[model_display_name] = []
                        model_display_names.append(model_display_name)
                    all_predictions[model_display_name].extend([0] * len(y_test))
            else:
                print(f"  ⚠️  {model_display_name} checkpoint not found")
                if model_display_name not in all_predictions:
                    all_predictions[model_display_name] = []
                    model_display_names.append(model_display_name)
                all_predictions[model_display_name].extend([0] * len(y_test))
    
    # Remove models that weren't found in any fold
    model_display_names = [name for name in model_display_names if len(all_predictions[name]) > 0]
    
    if len(model_display_names) == 0:
        print("\n✗ ERROR: No model checkpoints found!")
        print("Please run benchmark.py and model.py first.")
        return
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    for model_name in model_display_names:
        all_predictions[model_name] = np.array(all_predictions[model_name])
    
    # Create correctness matrix: 1 = correct, 0 = incorrect
    n_samples = len(all_labels)
    n_models = len(model_display_names)
    correctness_matrix = np.zeros((n_samples, n_models))
    
    for i, model_name in enumerate(model_display_names):
        correctness_matrix[:, i] = (all_predictions[model_name] == all_labels).astype(int)
    
    # Sort rows by LightGBM within each fold
    # This maintains fold grouping while ordering samples by LightGBM performance
    if "LightGBM" in model_display_names:
        lightgbm_idx = model_display_names.index("LightGBM")
        
        # Sort within each fold
        sorted_correctness = []
        sorted_labels = []
        new_fold_boundaries = [0]
        
        for fold_idx in range(n_folds):
            fold_start = fold_boundaries[fold_idx]
            fold_end = fold_boundaries[fold_idx + 1]
            
            # Get data for this fold
            fold_correctness = correctness_matrix[fold_start:fold_end]
            fold_labels = all_labels[fold_start:fold_end]
            
            # Sort by LightGBM column (0s first, then 1s)
            fold_sort_indices = np.argsort(fold_correctness[:, lightgbm_idx])
            
            sorted_correctness.append(fold_correctness[fold_sort_indices])
            sorted_labels.append(fold_labels[fold_sort_indices])
            new_fold_boundaries.append(new_fold_boundaries[-1] + len(fold_sort_indices))
        
        correctness_matrix = np.vstack(sorted_correctness)
        all_labels = np.concatenate(sorted_labels)
        fold_boundaries = new_fold_boundaries
        
        print("  ℹ️  Rows sorted by LightGBM predictions within each fold (incorrect samples first)")
    
    # Calculate statistics
    print(f"\n{'='*80}")
    print("CARPET PLOT STATISTICS")
    print(f"{'='*80}")
    
    # Per-model accuracy
    print("\nPer-Model Accuracy:")
    for i, model_name in enumerate(model_display_names):
        acc = correctness_matrix[:, i].mean()
        print(f"  {model_name:20s}: {acc:.3f}")
    
    # Samples that all models get correct/wrong
    all_correct = np.all(correctness_matrix == 1, axis=1)
    all_wrong = np.all(correctness_matrix == 0, axis=1)
    print(f"\nSamples all models get correct: {all_correct.sum()} ({100*all_correct.mean():.1f}%)")
    print(f"Samples all models get wrong:   {all_wrong.sum()} ({100*all_wrong.mean():.1f}%)")
    
    # Complementarity: samples where at least one model is correct
    at_least_one_correct = np.any(correctness_matrix == 1, axis=1)
    print(f"Samples at least one model gets correct: {at_least_one_correct.sum()} ({100*at_least_one_correct.mean():.1f}%)")
    print(f"  → Ensemble potential improvement: {100*(at_least_one_correct.mean() - correctness_matrix.mean()):.1f}%")
    
    # Create the carpet plot
    print(f"\n{'='*80}")
    print("GENERATING PLOT")
    print(f"{'='*80}")
    
    fig, ax = plt.subplots(figsize=(16, max(10, n_samples * 0.02)))
    
    # Create color map: green = correct (1), red = incorrect (0)
    cmap = plt.cm.colors.ListedColormap(['#d32f2f', '#388e3c'])  # Red, Green
    
    # Plot heatmap
    im = ax.imshow(correctness_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Add fold boundaries
    for boundary in fold_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=2, linestyle='--', alpha=0.7)
    
    # Add fold labels on the right
    for i in range(n_folds):
        mid_point = (fold_boundaries[i] + fold_boundaries[i+1]) / 2
        ax.text(n_models + 0.3, mid_point, f'Fold {i}', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Sample (grouped by fold)', fontsize=12, fontweight='bold')
    ax.set_title('Carpet Plot: Prediction Correctness Across Models and Folds\n' + 
                 'Green = Correct | Red = Incorrect', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # X-axis: model names
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_display_names, rotation=45, ha='right')
    
    # Y-axis: sample indices (sparse labels)
    y_tick_positions = []
    y_tick_labels = []
    for i in range(0, n_samples, max(1, n_samples // 20)):  # Show ~20 labels
        y_tick_positions.append(i)
        y_tick_labels.append(str(i))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    
    # Legend (simplified, no colorbar)
    green_patch = mpatches.Patch(color='#388e3c', label='Correct')
    red_patch = mpatches.Patch(color='#d32f2f', label='Incorrect')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', bbox_to_anchor=(1.02, 1), 
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save
    output_path = PLOT_DIR / "carpet_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Carpet plot saved to: {output_path}")
    
    plt.close()
    
    # Also create a summary heatmap showing model agreement
    print(f"\n{'='*80}")
    print("GENERATING MODEL AGREEMENT MATRIX")
    print(f"{'='*80}")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Calculate pairwise agreement between models
    agreement_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            agreement = np.mean(all_predictions[model_display_names[i]] == all_predictions[model_display_names[j]])
            agreement_matrix[i, j] = agreement
    
    # Plot
    im = ax.imshow(agreement_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            text = ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Configure
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax.set_yticklabels(model_display_names)
    ax.set_title('Model Agreement Matrix\n(Fraction of test samples with same prediction)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.colorbar(im, ax=ax, label='Agreement')
    plt.tight_layout()
    
    output_path = PLOT_DIR / "model_agreement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Model agreement matrix saved to: {output_path}")
    
    plt.close()
    
    # Print wrong/total for each model
    print(f"\n{'='*80}")
    print("WRONG PREDICTIONS PER MODEL")
    print(f"{'='*80}")
    
    for i, model_name in enumerate(model_display_names):
        n_correct = int(correctness_matrix[:, i].sum())
        n_wrong = n_samples - n_correct
        accuracy = correctness_matrix[:, i].mean()
        print(f"  {model_name:20s}: {n_wrong:4d}/{n_samples:4d} wrong ({100*(1-accuracy):5.2f}% error rate)")
    
    print(f"\n{'='*80}")
    print("✓ CARPET PLOT GENERATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    create_carpet_plot()
