#!/usr/bin/env python3
"""
PPI Prediction Web App

A simple Flask web application for predicting protein-protein interactions
using the trained LightGBM model on ESM-2 embeddings.
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
import torch
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
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "model1"

# Global variables for model and tokenizer
model = None
tokenizer = None
esm_model = None

def load_lightgbm_model():
    """Load the best LightGBM model from checkpoints."""
    # Try to find any LightGBM checkpoint
    lightgbm_checkpoints = list(CHECKPOINTS_DIR.glob("fold_*_esm2_LightGBM.pkl"))
    
    if not lightgbm_checkpoints:
        raise FileNotFoundError(
            f"No LightGBM checkpoints found in {CHECKPOINTS_DIR}. "
            "Please run benchmark.py first to train the model."
        )
    
    # Load the first checkpoint (they should all be similar in performance)
    checkpoint_path = lightgbm_checkpoints[0]
    print(f"Loading LightGBM model from: {checkpoint_path}")
    return joblib.load(checkpoint_path)


def load_esm2_model():
    """Load ESM-2 model and tokenizer."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library not available")
    
    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading ESM-2 model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"ESM-2 model loaded on device: {device}")
    return tokenizer, model


def get_esm2_embedding(sequence: str) -> np.ndarray:
    """Get ESM-2 embedding for a protein sequence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        
        # Create pair features
        pair_features = create_pair_features(emb_a, emb_b)
        
        # Predict
        prediction_proba = model.predict_proba(pair_features)[0]
        prediction = int(model.predict(pair_features)[0])
        confidence = float(prediction_proba[prediction])
        
        # Format result
        result = {
            'prediction': 'YES' if prediction == 1 else 'NO',
            'confidence': round(confidence * 100, 2),
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
    print("="*80)
    
    # Load models
    try:
        model = load_lightgbm_model()
        print("✓ LightGBM model loaded (ROC-AUC: 0.8861)")
    except Exception as e:
        print(f"✗ Error loading LightGBM model: {e}")
        exit(1)
    
    try:
        tokenizer, esm_model = load_esm2_model()
        print("✓ ESM-2 model loaded")
    except Exception as e:
        print(f"✗ Error loading ESM-2 model: {e}")
        exit(1)
    
    print("\n" + "="*80)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5001")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

