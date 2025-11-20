#!/usr/bin/env python3
"""
Model 3a: LoRA Fine-Tuning of ProtBERT-BFD (Bi-Encoder)
=======================================================

This script fine-tunes ProtBERT-BFD using a Siamese (Bi-Encoder) architecture
with LoRA, optimized for Protein-Protein Interaction (PPI) prediction.

Model: Rostlab/prot_bert_bfd
Reference: https://huggingface.co/Rostlab/prot_bert_bfd

Architecture (Bi-Encoder):
1. Independent Encoding: SeqA -> ProtBERT -> EmbA, SeqB -> ProtBERT -> EmbB
2. Pair Representation: Concat(EmbA, EmbB, |EmbA-EmbB|, EmbA*EmbB)
3. Classification: MLP -> Probability

Key Configuration:
- Space-separated amino acids (REQUIRED for ProtBERT)
- Max length: 512 (per protein)
- Target modules: query, key (only)
- Learning Rate: 2e-5
"""

from __future__ import annotations

import argparse
import json
import pickle
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress tokenizer overflow warnings
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
# Suppress gradient checkpointing warning with LoRA
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

# Try to import transformers and PEFT
try:
    from transformers import AutoTokenizer, AutoModel, logging as hf_logging
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
RESULTS_DIR = PROJECT_ROOT
PLOT_DIR = PROJECT_ROOT / "plot"
PLOT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "lora_protbert_predictions.txt"

# UPDATED MODEL CONFIGURATION
MODEL_NAME = "Rostlab/prot_bert_bfd"
MODEL_DISPLAY_NAME = "ProtBERT-BFD (Bi-Encoder)"


class ProtBERTBiEncoder(nn.Module):
    """
    Siamese Bi-Encoder architecture for PPI prediction.
    Wraps a LoRA-adapted ProtBERT model.
    """
    def __init__(self, base_model, hidden_dim=1024):
        super().__init__()
        self.encoder = base_model
        
        # Interaction Classifier MLP
        # Input: 4 * hidden_dim (concat, diff, product) -> 4096 dims for ProtBERT
        input_dim = hidden_dim * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # Encode Protein A
        # Use last_hidden_state[:, 0, :] for [CLS] token embedding
        out_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
        emb_a = out_a.last_hidden_state[:, 0, :]
        
        # Encode Protein B
        out_b = self.encoder(input_ids=input_ids_b, attention_mask=attention_mask_b)
        emb_b = out_b.last_hidden_state[:, 0, :]
        
        # Combine embeddings
        diff = torch.abs(emb_a - emb_b)
        prod = emb_a * emb_b
        pair_vector = torch.cat([emb_a, emb_b, diff, prod], dim=1)
        
        # Classify
        logits = self.classifier(pair_vector)
        return logits


class PPIDataset(Dataset):
    """Dataset for protein-protein interaction pairs (Bi-Encoder format)."""
    
    def __init__(self, pairs: List[Tuple[str, str, str, str]], labels: np.ndarray, tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # ProtBERT ALWAYS requires spaces
        self.add_spaces = True
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        id_a, id_b, seq_a, seq_b = self.pairs[idx]
        label = int(self.labels[idx])
        
        # ProtBERT requires spaces between amino acids
        if self.add_spaces:
            seq_a = " ".join(list(seq_a))
            seq_b = " ".join(list(seq_b))
        
        # Tokenize independently
        encoded_a = self.tokenizer(
            seq_a,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        encoded_b = self.tokenizer(
            seq_b,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': encoded_a['input_ids'].squeeze(0),
            'attention_mask_a': encoded_a['attention_mask'].squeeze(0),
            'input_ids_b': encoded_b['input_ids'].squeeze(0),
            'attention_mask_b': encoded_b['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float) # Float for BCEWithLogitsLoss
        }


def load_model_and_tokenizer(device: str = "cpu"):
    """Load ProtBERT-BFD and tokenizer with LoRA configuration."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library not installed.")
    if not PEFT_AVAILABLE:
        raise ImportError("peft library not installed.")
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"  Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    
    # Load base encoder (AutoModel, NOT AutoModelForSequenceClassification)
    base_encoder = AutoModel.from_pretrained(MODEL_NAME)
    
    # Configure LoRA for the Encoder
    # ProtBERT LoRA officially uses only Q,K
    target_modules = ["query", "key"]
    print(f"  ✓ Using target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=8,                    # Increased rank slightly for better capacity
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION 
    )
    
    # Apply LoRA to encoder
    encoder = get_peft_model(base_encoder, lora_config)
    
    # Create Bi-Encoder Model
    model = ProtBERTBiEncoder(encoder, hidden_dim=1024)
    
    # Freeze embeddings
    for name, param in model.named_parameters():
        if "embeddings" in name and "position" not in name:
             param.requires_grad = False
    
    # Ensure Classifier and LoRA params are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total trainable parameters: {trainable_params:,}")
    
    model.to(device)
    return model, tokenizer


def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, verbose=True, gradient_accumulation_steps=1):
    """Train the Bi-Encoder model."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Use BCEWithLogitsLoss for binary classification (safer than Sigmoid+BCELoss)
    criterion = nn.BCEWithLogitsLoss()
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.1)
    
    best_val_loss = float('inf')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("  Training...")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move inputs to device
            input_ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1) # [batch, 1]
            
            logits = model(input_ids_a, mask_a, input_ids_b, mask_b)
            loss = criterion(logits, labels)
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            train_loss += loss.item() * gradient_accumulation_steps
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if verbose and (batch_idx + 1) % 10 == 0:
                progress = batch_idx + 1
                percentage = (progress / len(train_loader)) * 100
                avg_loss = train_loss / progress
                print(f"\r    Batch {progress}/{len(train_loader)} ({percentage:.1f}%) | Loss: {avg_loss:.4f}", end='', flush=True)
        
        if verbose:
            print()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids_a = batch['input_ids_a'].to(device)
                mask_a = batch['attention_mask_a'].to(device)
                input_ids_b = batch['input_ids_b'].to(device)
                mask_b = batch['attention_mask_b'].to(device)
                labels = batch['labels'].to(device).unsqueeze(1)
                
                logits = model(input_ids_a, mask_a, input_ids_b, mask_b)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Metrics
        y_true = np.array(all_labels)
        y_pred_probs = np.array(all_preds)
        y_pred = (y_pred_probs >= 0.5).astype(int)
        val_acc = accuracy_score(y_true, y_pred)
        val_auc = roc_auc_score(y_true, y_pred_probs)
        
        if verbose:
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

# Helper functions for data loading (reused from previous implementation)
def load_curated_data(fold_idx: int = None):
    if not CURATED_DATA_DIR.exists():
        raise FileNotFoundError(f"Directory {CURATED_DATA_DIR} not found.")
    
    if fold_idx is not None:
        data_dir = CURATED_DATA_DIR / f"fold_{fold_idx}"
        metadata_path = data_dir / "split_info.json"
    else:
        data_dir = CURATED_DATA_DIR
        metadata_path = CURATED_DATA_DIR / "global_metadata.json" # Fallback
        
    if not metadata_path.exists():
        # Try to find split_info in fold_0 if generic one missing
        if (CURATED_DATA_DIR / "fold_0/split_info.json").exists():
             metadata_path = CURATED_DATA_DIR / "fold_0/split_info.json"

    with metadata_path.open('r') as f:
        metadata = json.load(f)
        
    y_train = np.load(data_dir / "train_labels.npy")
    y_test = np.load(data_dir / "test_labels.npy")
    
    with (data_dir / "train_pairs.pkl").open('rb') as f:
        train_pairs = pickle.load(f)
    with (data_dir / "test_pairs.pkl").open('rb') as f:
        test_pairs = pickle.load(f)
        
    return {
        'y_train': y_train,
        'y_test': y_test,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
    }

def get_available_folds():
    folds = []
    for i in range(10):
        if (CURATED_DATA_DIR / f"fold_{i}/train_labels.npy").exists():
            folds.append(i)
    return sorted(folds)

def evaluate_predictions(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    print(f"    Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"    F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"    ROC-AUC:   {roc_auc_score(y_true, y_pred_probs):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5) # Lowered LR
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512) # Set to 512
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-folds", type=int, default=1)
    args = parser.parse_args()
    
    print("="*80)
    print(f"MODEL 3a: {MODEL_DISPLAY_NAME}")
    print("="*80)
    
    folds = get_available_folds()[:args.n_folds]
    device = torch.device(args.device)
    
    for fold in folds:
        print(f"\nPROCESSING FOLD {fold}")
        data = load_curated_data(fold)
        
        model, tokenizer = load_model_and_tokenizer(args.device)
        
        train_ds = PPIDataset(data['train_pairs'], data['y_train'], tokenizer, args.max_length)
        # Use Test set as validation to monitor performance
        val_ds = PPIDataset(data['test_pairs'], data['y_test'], tokenizer, args.max_length)
        
        # Balanced Sampler
        pos = np.sum(data['y_train'])
        neg = len(data['y_train']) - pos
        weights = [1/neg if y==0 else 1/pos for y in data['y_train']]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, 
                             num_workers=args.num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
        
        train_model(model, train_dl, val_dl, device, args.epochs, args.learning_rate)
        
        # Final cleanup
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
