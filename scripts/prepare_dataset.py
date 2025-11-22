#!/usr/bin/env python3
"""
Dataset preparation script for PPI prediction with PROTEIN-DISJOINT splits.

This script:
1. Loads positive and negative protein pairs from FASTA files
2. Extracts ESM-2 embeddings and handcrafted features for each protein
3. Creates PROTEIN-DISJOINT stratified K-fold cross-validation splits
4. Saves all features and splits to curated_data/ for reuse

CRITICAL: PROTEIN-DISJOINT SPLITTING
=====================================
This script implements protein-disjoint splitting to prevent transductive leakage.

Standard (incorrect) approach: Split pairs randomly
  ❌ Same proteins appear in train and test (with different partners)
  ❌ Model sees protein features during training, tests on same proteins
  ❌ Inflates performance metrics, unrealistic for real-world deployment

Protein-disjoint (correct) approach: Split proteins first, then assign pairs
  ✅ Proteins are split into train/test sets
  ✅ Train pairs = both proteins in train set
  ✅ Test pairs = both proteins in test set
  ✅ No protein appears in both train and test
  ✅ Tests true generalization to unseen proteins
  ✅ Biologically realistic evaluation

This is the gold standard for PPI prediction evaluation and will be accepted
in peer review and industry interviews.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Bio.Data import IUPACData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold

warnings.filterwarnings("ignore")

# Amino acid order (21 residues including non-standard like selenocysteine 'U')
AA_ORDER = sorted(list(IUPACData.protein_letters) + ['U'])

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
ESM_CACHE_PATH = CACHE_DIR / "esm_embeddings.pkl"
HANDCRAFT_CACHE_PATH = CACHE_DIR / "handcrafted_features.pkl"


def sequence_hash(seq: str) -> str:
    """Compute SHA-256 hash of a sequence for cache keying."""
    return hashlib.sha256(seq.encode('utf-8')).hexdigest()


def load_feature_cache(path: Path) -> Dict:
    """Load feature cache from disk."""
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    return {}


def save_feature_cache(cache: Dict, path: Path):
    """Save feature cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(cache, f)


def parse_fasta(path: Path) -> List[Tuple[str, str, str, str]]:
    """
    Parse FASTA file and return list of (id_a, id_b, seq_a, seq_b) tuples.
    
    Args:
        path: Path to FASTA file
        
    Returns:
        List of tuples containing protein pair information
    """
    pairs = []
    header = None
    seq_lines = []
    
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    # Process previous entry
                    seq = "".join(seq_lines).replace(" ", "")
                    if "-" in seq:
                        seq_a, seq_b = seq.split("-", 1)
                        ids = header.split()
                        if len(ids) == 2:
                            pairs.append((ids[0], ids[1], seq_a, seq_b))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        
        # Process last entry
        if header is not None:
            seq = "".join(seq_lines).replace(" ", "")
            if "-" in seq:
                seq_a, seq_b = seq.split("-", 1)
                ids = header.split()
                if len(ids) == 2:
                    pairs.append((ids[0], ids[1], seq_a, seq_b))
    
    return pairs


def extract_handcrafted_features(seq: str) -> np.ndarray:
    """
    Extract handcrafted features from a protein sequence with SEGMENTED COMPOSITION.
    
    Features (75 total):
    - Splits sequence into 3 segments: N-terminus, Middle, C-terminus
    - For each segment, calculates 25 features:
      - Amino acid composition (21 features)
      - Sequence length (1 feature, relative to total)
      - Molecular weight (1 feature)
      - Average hydrophobicity (1 feature)
      - Net charge at pH 7.0 (1 feature)
    - Total: 25 * 3 = 75 features per protein
    
    Args:
        seq: Protein sequence string
        
    Returns:
        Feature vector of length 75
    """
    seq = seq.upper()
    if not seq:
        return np.zeros(75, dtype=float)
    
    length = len(seq)
    
    # Define segments
    # If sequence is too short (< 3), duplicate it across segments
    if length < 3:
        segments = [seq, seq, seq]
    else:
        # Split into 3 roughly equal parts
        part_len = length // 3
        # Handle remainders by adding to middle or last
        # Simple split: 0:L/3, L/3:2L/3, 2L/3:end
        s1 = seq[:part_len]
        s2 = seq[part_len:2*part_len]
        s3 = seq[2*part_len:]
        segments = [s1, s2, s3]
    
    all_features = []
    
    for segment in segments:
        # Count all 21 amino acids from original sequence (including 'U')
        aa_counts = {}
        valid_length = 0
        for aa in segment:
            if aa in AA_ORDER:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
                valid_length += 1
        
        if valid_length == 0:
            # Empty or invalid segment
            feats = np.zeros(25, dtype=float)
        else:
            # For Biopython analysis, use only standard 20 amino acids (remove 'U')
            cleaned_seq = "".join(aa for aa in segment if aa in IUPACData.protein_letters)
            if cleaned_seq:
                analysis = ProteinAnalysis(cleaned_seq)
                mw = analysis.molecular_weight()
                avg_hydrophobicity = analysis.gravy()
                net_charge = analysis.charge_at_pH(7.0)
            else:
                # Fallback if no standard amino acids
                mw = 0.0
                avg_hydrophobicity = 0.0
                net_charge = 0.0
            
            # Amino acid composition (21 features)
            aa_composition = np.array([aa_counts.get(aa, 0) / valid_length for aa in AA_ORDER], dtype=float)
            
            # Relative length (fraction of total protein)
            rel_len = valid_length / length if length > 0 else 0
            
            feats = np.concatenate([
                aa_composition,
                [rel_len],
                [mw],
                [avg_hydrophobicity],
                [net_charge],
            ])
            
        all_features.append(feats)
    
    # Concatenate all segment features
    return np.concatenate(all_features)


def get_esm2_embeddings(sequences: List[str], model, tokenizer, device: str = "cpu") -> np.ndarray:
    """
    Extract ESM-2 embeddings for a list of protein sequences.
    
    Args:
        sequences: List of protein sequences
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        device: Device to run model on
        
    Returns:
        Array of shape (n_sequences, 1280) containing mean-pooled embeddings
    """
    import torch
    
    embeddings = []
    model.eval()
    
    for i, seq in enumerate(sequences):
        if (i + 1) % 100 == 0:
            print(f"  Processing sequence {i+1}/{len(sequences)}...")
        
        # Tokenize
        encoded = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state[0, 1:-1, :]  # Remove [CLS] and [EOS]
            embedding = hidden.mean(dim=0).cpu().numpy()
        
        embeddings.append(embedding)
    
    return np.array(embeddings)


def ensure_handcrafted_features(pairs: List[Tuple[str, str, str, str]]) -> Dict[str, np.ndarray]:
    """
    Extract or load handcrafted features for all proteins in pairs.
    Uses cache to avoid recomputation.
    
    Args:
        pairs: List of (id_a, id_b, seq_a, seq_b) tuples
        
    Returns:
        Dictionary mapping protein ID to feature vector
    """
    cache = load_feature_cache(HANDCRAFT_CACHE_PATH)
    feature_lookup = {}
    
    # Collect unique proteins
    protein_data = {}  # id -> sequence
    for id_a, id_b, seq_a, seq_b in pairs:
        protein_data[id_a] = seq_a
        protein_data[id_b] = seq_b
    
    missing_ids = []
    for pid, seq in protein_data.items():
        seq_hash = sequence_hash(seq)
        if pid in cache and cache[pid].get('hash') == seq_hash:
            feature_lookup[pid] = cache[pid]['features']
        else:
            missing_ids.append(pid)
    
    if missing_ids:
        print(f"Computing handcrafted features for {len(missing_ids)} proteins...")
        for i, pid in enumerate(missing_ids):
            if (i + 1) % 100 == 0:
                print(f"  Processing protein {i+1}/{len(missing_ids)}...")
            seq = protein_data[pid]
            features = extract_handcrafted_features(seq)
            feature_lookup[pid] = features
            cache[pid] = {'hash': sequence_hash(seq), 'features': features}
        
        save_feature_cache(cache, HANDCRAFT_CACHE_PATH)
        print(f"Saved handcrafted features cache to {HANDCRAFT_CACHE_PATH}")
    else:
        print("All handcrafted features loaded from cache")
    
    return feature_lookup


def ensure_esm_embeddings(pairs: List[Tuple[str, str, str, str]], model, tokenizer, device: str = "cpu") -> Dict[str, np.ndarray]:
    """
    Extract or load ESM-2 embeddings for all proteins in pairs.
    Uses cache to avoid recomputation.
    
    Args:
        pairs: List of (id_a, id_b, seq_a, seq_b) tuples
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        device: Device to run model on
        
    Returns:
        Dictionary mapping protein ID to embedding vector
    """
    cache = load_feature_cache(ESM_CACHE_PATH)
    embedding_lookup = {}
    
    # Collect unique proteins
    protein_data = {}  # id -> sequence
    for id_a, id_b, seq_a, seq_b in pairs:
        protein_data[id_a] = seq_a
        protein_data[id_b] = seq_b
    
    missing_ids = []
    missing_seqs = []
    for pid, seq in protein_data.items():
        seq_hash = sequence_hash(seq)
        if pid in cache and cache[pid].get('hash') == seq_hash:
            embedding_lookup[pid] = cache[pid]['embedding']
        else:
            missing_ids.append(pid)
            missing_seqs.append(seq)
    
    if missing_ids:
        print(f"Computing ESM-2 embeddings for {len(missing_ids)} proteins...")
        embeddings = get_esm2_embeddings(missing_seqs, model, tokenizer, device)
        
        for pid, embedding in zip(missing_ids, embeddings):
            embedding_lookup[pid] = embedding
            cache[pid] = {'hash': sequence_hash(protein_data[pid]), 'embedding': embedding}
        
        save_feature_cache(cache, ESM_CACHE_PATH)
        print(f"Saved ESM-2 embeddings cache to {ESM_CACHE_PATH}")
    else:
        print("All ESM-2 embeddings loaded from cache")
    
    return embedding_lookup


def combine_pair_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Combine two protein embeddings into a pair feature vector.
    Uses: concat(embA, embB, abs(embA - embB), embA * embB)
    """
    diff = np.abs(emb_a - emb_b)
    product = emb_a * emb_b
    return np.concatenate([emb_a, emb_b, diff, product])


def prepare_pair_features(pairs: List[Tuple[str, str, str, str]], feature_lookup: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create pair feature matrix from protein feature lookup.
    
    Args:
        pairs: List of (id_a, id_b, seq_a, seq_b) tuples
        feature_lookup: Dictionary mapping protein ID to feature vector
        
    Returns:
        Feature matrix of shape (n_pairs, feature_dim)
    """
    features = []
    for id_a, id_b, _, _ in pairs:
        feat_a = feature_lookup[id_a]
        feat_b = feature_lookup[id_b]
        combined = combine_pair_features(feat_a, feat_b)
        features.append(combined)
    return np.array(features)


def save_fold_data(
    fold_dir: Path,
    pairs_train: List[Tuple[str, str, str, str]],
    pairs_test: List[Tuple[str, str, str, str]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_handcraft: np.ndarray,
    X_test_handcraft: np.ndarray,
    X_train_esm: np.ndarray = None,
    X_test_esm: np.ndarray = None,
    metadata: Dict = None
):
    """Save all data for a single fold to its directory."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    with (fold_dir / "train_pairs.pkl").open('wb') as f:
        pickle.dump(pairs_train, f)
    with (fold_dir / "test_pairs.pkl").open('wb') as f:
        pickle.dump(pairs_test, f)
    
    # Save labels
    np.save(fold_dir / "train_labels.npy", y_train)
    np.save(fold_dir / "test_labels.npy", y_test)
    
    # Save handcrafted features
    np.save(fold_dir / "train_features_handcrafted.npy", X_train_handcraft)
    np.save(fold_dir / "test_features_handcrafted.npy", X_test_handcraft)
    
    # Save ESM-2 features (if provided)
    if X_train_esm is not None and X_test_esm is not None:
        np.save(fold_dir / "train_features_esm2.npy", X_train_esm)
        np.save(fold_dir / "test_features_esm2.npy", X_test_esm)
    
    # Save metadata
    if metadata is not None:
        with (fold_dir / "split_info.json").open('w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare PPI dataset with protein-disjoint K-fold cross-validation splits")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--positive",
        type=Path,
        default=PROJECT_ROOT / "ppi_human_interactions.fasta",
        help="Path to positive PPI FASTA file",
    )
    parser.add_argument(
        "--negative",
        type=Path,
        default=PROJECT_ROOT / "ppi_negative_interactions.fasta",
        help="Path to negative PPI FASTA file",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of folds for cross-validation (default: 3, reduced from 5 due to stricter protein-disjoint splitting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for ESM-2 model (default: cpu)",
    )
    parser.add_argument(
        "--skip-esm",
        action="store_true",
        help="Skip ESM-2 embedding extraction (only prepare handcrafted features)",
    )
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    CURATED_DATA_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print(f"DATASET PREPARATION FOR PPI PREDICTION - {args.n_folds}-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: Loading FASTA files...")
    print("="*80)
    positive_pairs = parse_fasta(args.positive)
    print(f"   ✓ Loaded {len(positive_pairs)} positive pairs")
    
    negative_pairs = parse_fasta(args.negative)
    print(f"   ✓ Loaded {len(negative_pairs)} negative pairs")
    
    # Combine and create labels
    all_pairs = positive_pairs + negative_pairs
    labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))
    
    print(f"\n   Total pairs: {len(all_pairs)}")
    print(f"   Positive: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")
    
    # Extract handcrafted features (once for all folds)
    print("\n" + "="*80)
    print("STEP 2: Extracting handcrafted features (shared across all folds)...")
    print("="*80)
    handcraft_lookup = ensure_handcrafted_features(all_pairs)
    print("   ✓ Handcrafted features ready for all proteins")
    
    # Extract ESM-2 embeddings (once for all folds, if not skipped)
    esm_lookup = None
    if not args.skip_esm:
        print("\n" + "="*80)
        print("STEP 3: Extracting ESM-2 embeddings (shared across all folds)...")
        print("="*80)
        try:
            import torch
            from transformers import EsmModel, EsmTokenizer
            
            print("   Loading ESM-2 model...")
            model_name = "facebook/esm2_t33_650M_UR50D"
            tokenizer = EsmTokenizer.from_pretrained(model_name)
            model = EsmModel.from_pretrained(model_name)
            model.to(args.device)
            print("   ✓ ESM-2 model loaded")
            
            esm_lookup = ensure_esm_embeddings(all_pairs, model, tokenizer, args.device)
            print("   ✓ ESM-2 embeddings ready for all proteins")
            
        except Exception as e:
            print(f"   ✗ ERROR: Could not load ESM-2 model: {e}")
            print("   Skipping ESM-2 embeddings. Use --skip-esm to suppress this error.")
            args.skip_esm = True
    
    # Create 5-fold cross-validation splits (PROTEIN-DISJOINT)
    print("\n" + "="*80)
    print(f"STEP 4: Creating {args.n_folds}-fold PROTEIN-DISJOINT cross-validation splits...")
    print("="*80)
    print("   ℹ️  Using protein-disjoint splitting to prevent transductive leakage")
    print("   ℹ️  No protein will appear in both train and test sets")
    
    # Extract all unique proteins
    all_proteins = set()
    for id_a, id_b, _, _ in all_pairs:
        all_proteins.add(id_a)
        all_proteins.add(id_b)
    proteins_list = sorted(list(all_proteins))
    
    print(f"   ✓ Found {len(proteins_list)} unique proteins")
    
    # Create stratification labels for proteins based on their interaction patterns
    # This ensures each fold gets a balanced mix of proteins with different
    # positive/negative interaction ratios, maintaining class balance at pair level
    protein_pos_count = {p: 0 for p in proteins_list}
    protein_neg_count = {p: 0 for p in proteins_list}
    
    for (id_a, id_b, _, _), label in zip(all_pairs, labels):
        if label == 1:
            protein_pos_count[id_a] += 1
            protein_pos_count[id_b] += 1
        else:
            protein_neg_count[id_a] += 1
            protein_neg_count[id_b] += 1
    
    # Stratify proteins by their positive interaction ratio (binned)
    # This helps maintain pair-level class balance after protein-disjoint splitting
    protein_strata = []
    for p in proteins_list:
        total = protein_pos_count[p] + protein_neg_count[p]
        pos_ratio = protein_pos_count[p] / total if total > 0 else 0
        # Bin into 4 categories: 0-25%, 25-50%, 50-75%, 75-100%
        stratum = int(pos_ratio * 4)
        if stratum == 4:
            stratum = 3  # Handle edge case of 100%
        protein_strata.append(stratum)
    
    protein_strata = np.array(protein_strata)
    print(f"   ✓ Proteins stratified by interaction patterns")
    
    # Split proteins (not pairs!) into folds
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    protein_fold_indices = list(skf.split(proteins_list, protein_strata))
    
    # Build pair-level folds from protein-level splits
    fold_indices = []
    total_train_pairs = 0
    total_test_pairs = 0
    discarded_pairs = 0
    
    for fold_idx, (train_prot_idx, test_prot_idx) in enumerate(protein_fold_indices):
        train_proteins = set(np.array(proteins_list)[train_prot_idx])
        test_proteins = set(np.array(proteins_list)[test_prot_idx])
        
        # Verify no overlap
        assert len(train_proteins & test_proteins) == 0, "Protein leakage detected!"
        
        # Assign pairs to train/test based on BOTH proteins being in that set
        train_pair_indices = []
        test_pair_indices = []
        
        for i, (id_a, id_b, _, _) in enumerate(all_pairs):
            if id_a in train_proteins and id_b in train_proteins:
                train_pair_indices.append(i)
            elif id_a in test_proteins and id_b in test_proteins:
                test_pair_indices.append(i)
            # Pairs with one protein in train and one in test are discarded
        
        fold_indices.append((np.array(train_pair_indices), np.array(test_pair_indices)))
        total_train_pairs += len(train_pair_indices)
        total_test_pairs += len(test_pair_indices)
    
    discarded_pairs = len(all_pairs) - total_train_pairs - total_test_pairs
    
    print(f"   ✓ Created {args.n_folds} protein-disjoint folds")
    print(f"   Random seed: {args.seed}")
    print(f"   Total usable pairs across all folds: {total_train_pairs + total_test_pairs}/{len(all_pairs)}")
    if discarded_pairs > 0:
        print(f"   ⚠️  Discarded {discarded_pairs} cross-fold pairs (protein in train, partner in test)")
        print(f"   ℹ️  This is CORRECT behavior for protein-disjoint splits")
    
    # Process each fold
    print("\n" + "="*80)
    print(f"STEP 5: Processing {args.n_folds} folds...")
    print("="*80)
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        print(f"\n{'─'*80}")
        print(f"PROCESSING FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'─'*80}")
        
        # Get pairs and labels for this fold
        pairs_train = [all_pairs[i] for i in train_idx]
        pairs_test = [all_pairs[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        print(f"   Train set: {len(pairs_train)} pairs ({np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg)")
        print(f"   Test set: {len(pairs_test)} pairs ({np.sum(y_test == 1)} pos, {np.sum(y_test == 0)} neg)")
        
        # Create feature matrices for this fold
        print(f"   Creating feature matrices...")
        X_train_handcraft = prepare_pair_features(pairs_train, handcraft_lookup)
        X_test_handcraft = prepare_pair_features(pairs_test, handcraft_lookup)
        print(f"   ✓ Handcrafted features: train {X_train_handcraft.shape}, test {X_test_handcraft.shape}")
        
        X_train_esm = None
        X_test_esm = None
        if esm_lookup is not None:
            X_train_esm = prepare_pair_features(pairs_train, esm_lookup)
            X_test_esm = prepare_pair_features(pairs_test, esm_lookup)
            print(f"   ✓ ESM-2 features: train {X_train_esm.shape}, test {X_test_esm.shape}")
        
        # Count unique proteins in train and test
        train_proteins_in_fold = set()
        test_proteins_in_fold = set()
        for id_a, id_b, _, _ in pairs_train:
            train_proteins_in_fold.add(id_a)
            train_proteins_in_fold.add(id_b)
        for id_a, id_b, _, _ in pairs_test:
            test_proteins_in_fold.add(id_a)
            test_proteins_in_fold.add(id_b)
        
        # Verify no protein overlap (critical!)
        protein_overlap = train_proteins_in_fold & test_proteins_in_fold
        if protein_overlap:
            raise ValueError(f"Protein leakage detected in fold {fold_idx}: {len(protein_overlap)} proteins in both train and test!")
        
        # Create metadata for this fold
        fold_metadata = {
            "fold": fold_idx,
            "timestamp": datetime.utcnow().isoformat(),
            "positive_file": str(args.positive),
            "negative_file": str(args.negative),
            "seed": args.seed,
            "split_strategy": "protein_disjoint",
            "split_description": "No protein appears in both train and test sets (prevents transductive leakage)",
            "n_total": len(all_pairs),
            "n_positive": int(np.sum(labels == 1)),
            "n_negative": int(np.sum(labels == 0)),
            "n_train": len(pairs_train),
            "n_test": len(pairs_test),
            "train_positive": int(np.sum(y_train == 1)),
            "train_negative": int(np.sum(y_train == 0)),
            "test_positive": int(np.sum(y_test == 1)),
            "test_negative": int(np.sum(y_test == 0)),
            "n_unique_proteins_train": len(train_proteins_in_fold),
            "n_unique_proteins_test": len(test_proteins_in_fold),
            "n_unique_proteins_total": len(train_proteins_in_fold) + len(test_proteins_in_fold),
            "protein_overlap_count": len(protein_overlap),
            "handcrafted_feature_dim": X_train_handcraft.shape[1],
            "esm2_available": not args.skip_esm,
        }
        
        if not args.skip_esm:
            fold_metadata["esm2_feature_dim"] = X_train_esm.shape[1]
        
        # Save fold data
        fold_dir = CURATED_DATA_DIR / f"fold_{fold_idx}"
        print(f"   Saving fold data to: {fold_dir}/")
        save_fold_data(
            fold_dir=fold_dir,
            pairs_train=pairs_train,
            pairs_test=pairs_test,
            y_train=y_train,
            y_test=y_test,
            X_train_handcraft=X_train_handcraft,
            X_test_handcraft=X_test_handcraft,
            X_train_esm=X_train_esm,
            X_test_esm=X_test_esm,
            metadata=fold_metadata
        )
        
        print(f"   ✓ FOLD {fold_idx + 1}/{args.n_folds} COMPLETE!")
        print(f"   ────────────────────────────────────────────────────────────────")
    
    # Save global metadata
    print("\n" + "="*80)
    print("STEP 6: Saving global metadata...")
    print("="*80)
    
    global_metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "positive_file": str(args.positive),
        "negative_file": str(args.negative),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "split_strategy": "protein_disjoint",
        "split_description": "Proteins are split into folds first, then pairs are assigned based on protein membership. No protein appears in both train and test sets within any fold. This prevents transductive leakage and tests true generalization to unseen proteins.",
        "n_total": len(all_pairs),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "n_unique_proteins": len(proteins_list),
        "handcrafted_feature_dim": X_train_handcraft.shape[1],
        "esm2_available": not args.skip_esm,
    }
    
    if not args.skip_esm:
        global_metadata["esm2_feature_dim"] = X_train_esm.shape[1]
    
    with (CURATED_DATA_DIR / "global_metadata.json").open('w') as f:
        json.dump(global_metadata, f, indent=2)
    print(f"   ✓ Saved global metadata")
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n✅ PROTEIN-DISJOINT SPLITS (NO TRANSDUCTIVE LEAKAGE)")
    print(f"   All {args.n_folds} folds use protein-level splitting")
    print(f"   No protein appears in both train and test within any fold")
    print(f"   This ensures true generalization to unseen proteins")
    print(f"\nAll {args.n_folds} folds saved to: {CURATED_DATA_DIR}/")
    for i in range(args.n_folds):
        print(f"  - fold_{i}/")
    print(f"\nEach fold contains:")
    print(f"  - train_pairs.pkl, test_pairs.pkl")
    print(f"  - train_labels.npy, test_labels.npy")
    print(f"  - train_features_handcrafted.npy, test_features_handcrafted.npy")
    if not args.skip_esm:
        print(f"  - train_features_esm2.npy, test_features_esm2.npy")
    print(f"  - split_info.json")
    print(f"\nYou can now run models using any of these folds for cross-validation.")


if __name__ == "__main__":
    main()

