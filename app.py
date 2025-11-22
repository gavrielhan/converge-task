#!/usr/bin/env python3
"""
PPI Prediction Web App

A Flask web application for predicting protein-protein interactions
using an ensemble method: LightGBM + Model2B fallback.
Uses Model2B when LightGBM confidence < 0.7 threshold.
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import requests
from io import StringIO

# ESM-2 imports
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install with: pip install transformers")

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR_MODEL1 = PROJECT_ROOT / "checkpoints" / "model1"
CHECKPOINTS_DIR_MODEL2 = PROJECT_ROOT / "checkpoints" / "model2"

# Global variables for models and tokenizer
lgbm_model = None
model2b = None
tokenizer = None
esm_model = None
device = None
CONFIDENCE_THRESHOLD = 0.7  # Use Model2B when LightGBM confidence < 0.7


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

def load_lightgbm_model():
    """Load the best LightGBM model from checkpoints."""
    # Try to find any LightGBM checkpoint
    lightgbm_checkpoints = list(CHECKPOINTS_DIR_MODEL1.glob("fold_*_esm2_LightGBM.pkl"))
    
    if not lightgbm_checkpoints:
        raise FileNotFoundError(
            f"No LightGBM checkpoints found in {CHECKPOINTS_DIR_MODEL1}. "
            "Please run benchmark.py first to train the model."
        )
    
    # Load the first checkpoint (they should all be similar in performance)
    checkpoint_path = lightgbm_checkpoints[0]
    print(f"Loading LightGBM model from: {checkpoint_path}")
    return joblib.load(checkpoint_path)


def load_model2b(device):
    """Load Model2B from checkpoints."""
    # Try to find any Model2B checkpoint
    model2b_checkpoints = list(CHECKPOINTS_DIR_MODEL2.glob("fold_*_Model22B.pth"))
    
    if not model2b_checkpoints:
        raise FileNotFoundError(
            f"No Model2B checkpoints found in {CHECKPOINTS_DIR_MODEL2}. "
            "Please train Model2B first."
        )
    
    # Load the first checkpoint
    checkpoint_path = model2b_checkpoints[0]
    print(f"Loading Model2B from: {checkpoint_path}")
    
    model2b = Model2B_SiameseMLP(protein_emb_dim=1280)
    model2b.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model2b = model2b.to(device)
    model2b.eval()
    
    return model2b


def load_esm2_model(device):
    """Load ESM-2 model and tokenizer."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library not available")
    
    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading ESM-2 model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to GPU if available
    model = model.to(device)
    model.eval()
    
    print(f"ESM-2 model loaded on device: {device}")
    return tokenizer, model


def get_esm2_embedding(sequence: str) -> np.ndarray:
    """Get ESM-2 embedding for a protein sequence."""
    # Tokenize
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = esm_model(**inputs)
        # Use mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy().flatten()


def create_pair_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Create pair features from two protein embeddings."""
    # Concatenate: [emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b]
    diff = np.abs(emb_a - emb_b)
    product = emb_a * emb_b
    pair_features = np.concatenate([emb_a, emb_b, diff, product])
    return pair_features.reshape(1, -1)  # Shape: (1, feature_dim)


def fetch_protein_sequence_from_uniprot(gene_name: str) -> tuple:
    """
    Fetch protein sequence from UniProt by gene name.
    
    Returns:
        tuple: (sequence, uniprot_id, protein_name) or (None, None, None) if not found
    """
    # Try to search UniProt
    search_url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9606&format=json&size=1"
    
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                result = data['results'][0]
                uniprot_id = result.get('primaryAccession', '')
                protein_name = result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', gene_name)
                sequence = result.get('sequence', {}).get('value', '')
                
                if sequence:
                    return sequence, uniprot_id, protein_name
    except Exception as e:
        print(f"Error fetching from UniProt: {e}")
    
    return None, None, None


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict protein-protein interaction."""
    try:
        data = request.json
        input_a = data.get('protein_a', '').strip()
        input_b = data.get('protein_b', '').strip()
        
        if not input_a or not input_b:
            return jsonify({'error': 'Both protein inputs are required'}), 400
        
        # Determine if inputs are sequences or gene names
        sequence_a = None
        sequence_b = None
        name_a = "Protein A"
        name_b = "Protein B"
        
        # Check if input_a is a sequence (contains only valid amino acids)
        if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in input_a.upper()):
            sequence_a = input_a.upper()
            name_a = f"Custom sequence ({len(sequence_a)} aa)"
        else:
            # Try to fetch from UniProt
            sequence_a, uniprot_a, protein_name_a = fetch_protein_sequence_from_uniprot(input_a)
            if sequence_a:
                name_a = f"{input_a} ({uniprot_a})"
            else:
                return jsonify({'error': f'Could not find protein sequence for gene: {input_a}'}), 404
        
        # Check if input_b is a sequence
        if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in input_b.upper()):
            sequence_b = input_b.upper()
            name_b = f"Custom sequence ({len(sequence_b)} aa)"
        else:
            # Try to fetch from UniProt
            sequence_b, uniprot_b, protein_name_b = fetch_protein_sequence_from_uniprot(input_b)
            if sequence_b:
                name_b = f"{input_b} ({uniprot_b})"
            else:
                return jsonify({'error': f'Could not find protein sequence for gene: {input_b}'}), 404
        
        # Get ESM-2 embeddings
        print(f"Computing embeddings for {name_a} and {name_b}...")
        emb_a = get_esm2_embedding(sequence_a)
        emb_b = get_esm2_embedding(sequence_b)
        
        # Create pair features for LightGBM
        pair_features = create_pair_features(emb_a, emb_b)
        
        # Get LightGBM prediction
        lgbm_proba = lgbm_model.predict_proba(pair_features)[0]
        lgbm_pred = int(lgbm_model.predict(pair_features)[0])
        lgbm_confidence = float(np.max(lgbm_proba))
        
        # Determine which model to use
        model_used = 'LightGBM'
        prediction_proba = lgbm_proba
        prediction = lgbm_pred
        
        # Use Model2B if LightGBM confidence is below threshold
        if lgbm_confidence < CONFIDENCE_THRESHOLD:
            model_used = 'Model2B'
            print(f"  LightGBM confidence ({lgbm_confidence:.3f}) < threshold ({CONFIDENCE_THRESHOLD}), using Model2B...")
            
            # Prepare embeddings for Model2B
            emb_a_tensor = torch.FloatTensor(emb_a).unsqueeze(0).to(device)
            emb_b_tensor = torch.FloatTensor(emb_b).unsqueeze(0).to(device)
            
            # Predict with Model2B
            with torch.no_grad():
                model2b_output = model2b(emb_a_tensor, emb_b_tensor)
                model2b_prob_interact = float(model2b_output.cpu().numpy().flatten()[0])
            
            # Format Model2B probabilities
            prediction_proba = np.array([1 - model2b_prob_interact, model2b_prob_interact])
            prediction = 1 if model2b_prob_interact > 0.5 else 0
        else:
            print(f"  LightGBM confidence ({lgbm_confidence:.3f}) >= threshold ({CONFIDENCE_THRESHOLD}), using LightGBM")
        
        confidence = float(prediction_proba[prediction])
        
        # Format result
        result = {
            'prediction': 'YES' if prediction == 1 else 'NO',
            'confidence': round(confidence * 100, 2),
            'model_used': model_used,
            'lgbm_confidence': round(lgbm_confidence * 100, 2),
            'protein_a': {
                'name': name_a,
                'length': len(sequence_a)
            },
            'protein_b': {
                'name': name_b,
                'length': len(sequence_b)
            },
            'probability_interact': round(float(prediction_proba[1]) * 100, 2),
            'probability_no_interact': round(float(prediction_proba[0]) * 100, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*80)
    print("PPI PREDICTION WEB APP")
    print("Using Ensemble: LightGBM + Model2B Fallback (threshold=0.7)")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ESM-2 model first (needed for embeddings)
    try:
        tokenizer, esm_model = load_esm2_model(device)
        print("✓ ESM-2 model loaded")
    except Exception as e:
        print(f"✗ Error loading ESM-2 model: {e}")
        exit(1)
    
    # Load LightGBM model
    try:
        lgbm_model = load_lightgbm_model()
        print("✓ LightGBM model loaded")
    except Exception as e:
        print(f"✗ Error loading LightGBM model: {e}")
        exit(1)
    
    # Load Model2B model
    try:
        model2b = load_model2b(device)
        print("✓ Model2B loaded")
    except Exception as e:
        print(f"✗ Error loading Model2B: {e}")
        exit(1)
    
    print(f"\n✓ Ensemble ready: LightGBM (primary) + Model2B (fallback when confidence < {CONFIDENCE_THRESHOLD})")
    print(f"  Expected performance: ROC-AUC ~0.8879")
    
    print("\n" + "="*80)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5001")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

