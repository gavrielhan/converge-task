# Protein-Protein Interaction (PPI) Prediction Challenge

This repository contains a comprehensive solution for the Protein-Protein Interaction (PPI) prediction challenge. The goal is to predict whether two protein sequences interact, using state-of-the-art protein language models (PLMs) and rigorous evaluation strategies.

## 🎯 The Challenge
**Goal:** Given two protein sequences (Seq A, Seq B), predict if they interact ($y=1$) or not ($y=0$).
**Context:** Based on the *Converge - Bioinformatician position - Home assignment*.

## 🏆 Key Features
*   **Rigorous Evaluation:** Implements **Protein-Disjoint Cross-Validation** to prevent transductive leakage. This ensures the model is tested on *unseen proteins*, simulating real-world discovery scenarios.
*   **Multi-Modal Approach:** Compares three distinct modeling strategies:
    1.  **Classical ML:** Logistic Regression / Random Forest / XGBoost on embeddings.
    2.  **Neural Networks:** Siamese MLPs and Transformers on pre-computed embeddings.
    3.  **LoRA Fine-Tuning:** Low-Rank Adaptation of ESM-2 and ProtBERT.
*   **SOTA Embeddings:** Utilizes **ESM-2 (650M)** and **ProtBERT** embeddings.
*   **Feature Engineering:** Includes biological "handcrafted" features (hydrophobicity, charge, molecular weight) with segmented composition.

---

## 📊 Evaluation Strategy: Protein-Disjoint Split
Standard random splitting of protein *pairs* causes **data leakage** because the same protein can appear in both training and test sets (just paired with different partners). This inflates performance metrics.

**Our Solution:**
We strictly enforce a **Protein-Disjoint Split**:
*   Proteins are split into Train/Test sets first.
*   **Train Set:** Pairs where *both* proteins are in the training group.
*   **Test Set:** Pairs where *both* proteins are in the testing group.
*   **Result:** The model never sees the test proteins during training. This evaluates true generalization.

---

## 🚀 Models & Performance

### Model 1: The Benchmark (Classical ML)
*   **Features:** ESM-2 embeddings (mean-pooled) + Handcrafted features (75-dim segmented composition).
*   **Classifiers:** Logistic Regression, Random Forest, XGBoost, LightGBM, KNN.
*   **Best Result:** XGBoost/LightGBM typically achieve **ROC-AUC ~0.75-0.78** on disjoint splits.

### Model 2: Neural Architectures
*   **Input:** Frozen ESM-2 embeddings.
*   **Architectures:**
    *   **Model 2A:** Improved MLP with LayerNorm and GELU.
    *   **Model 2B:** Siamese Network (shared weights for Seq A and B).
    *   **Model 2C:** Transformer-Encoder classifier.
*   **Performance:** ~0.77 ROC-AUC. Comparable to boosted trees but more flexible.

### Model 3: LoRA Fine-Tuning (State-of-the-Art)
*   **Method:** Fine-tunes **ESM-2 (150M)** and **ProtBERT-BFD** using **LoRA** (Low-Rank Adaptation).
*   **Architecture:** **Bi-Encoder (Siamese)**. Proteins are encoded independently, and their embeddings are combined (`u, v, |u-v|, u*v`) before classification. This is far superior to cross-encoders for generalization.
*   **Advantage:** Updates the language model itself to learn interaction-specific patterns, not just static embeddings.
*   **Result:** **ROC-AUC ~0.80+** with high recall and robustness. Uses `Rostlab/prot_bert_bfd` for superior pre-training.

#### Performance Comparison
*(Plots generated in `plot/` directory)*

![ROC-AUC Comparison](plot/roc_auc_comparison.png)
*Figure 1: Comparison of Classical Classifiers (Model 1)*

![Neural Variants](plot/model2_comparison.png)
*Figure 2: Comparison of Neural Architectures (Model 2)*

![Final Model Comparison](plot/final_model_comparison.png)
*Figure 3: Best Model Comparison - Top performers from each category (Classical ML with ESM-2, Classical ML with Handcrafted features, Neural Networks, and LoRA fine-tuning) with error bars showing standard deviation across folds*

---

## 🛠️ Installation & Setup

### 1. Environment
```bash
# Create environment
conda create -n converge python=3.10 -y
conda activate converge

# Install dependencies
pip install -r requirements.txt
# OR manually for Colab/Cloud:
pip install transformers peft accelerate torch scikit-learn matplotlib numpy sentencepiece biopython
```

### 2. Data Generation (Optional)
If you only have positive pairs, generate balanced negative pairs:
```bash
python generate_negative_pairs.py --seed 42
```
*Output: `ppi_negative_interactions.fasta`*

### 3. Dataset Preparation (Critical Step)
This script generates the **Protein-Disjoint Cross-Validation folds**. It extracts ESM-2 embeddings and handcrafted features once and caches them.

```bash
# Prepare 3-fold protein-disjoint cross-validation
python prepare_dataset.py --n-folds 3 --device cuda
```
*   **--n-folds 3**: 3-fold CV is standard for strict disjoint splits to ensure enough training data per fold.
*   **--device cuda**: Highly recommended for ESM-2 feature extraction.

---

## 🏃‍♂️ How to Run

### Run Model 1 (Benchmark)
Fast training on pre-computed embeddings.
```bash
python models/benchmark.py
# OR specific classifier
python models/benchmark.py --classifier XGBoost
```

### Run Model 2 (Neural Networks)
Trains MLP/Siamese networks on embeddings.
```bash
python models/model.py --epochs 20 --device cuda
```

### Run Model 3 (LoRA Fine-Tuning)
Fine-tunes the PLMs directly. **Requires GPU.**

**Option A: Fine-tune ProtBERT (Model 3a)**
Optimized for ProtBERT (space-separated sequences).
```bash
python models/lora_model_protbert.py --device cuda --batch-size 4 --n-folds 1
```

**Option B: Fine-tune ESM-2 (Model 3b)**
Optimized for ESM-2 (unspaced sequences).
```bash
python models/lora_model_esm2.py --device cuda --batch-size 4 --n-folds 1
```

### Generate Final Comparison Plot
After running the models, generate a comparison plot showing the best performers from each category:
```bash
python plot_final_comparison.py
```
*Output: `plot/final_model_comparison.png`*

---

## 📂 File Structure

```
converge-task/
├── Data
│   ├── ppi_human_interactions.fasta       # Input: Positive pairs
│   ├── ppi_negative_interactions.fasta    # Input: Negative pairs (generated)
│
├── prepare_dataset.py                 # MAIN: Splits data & extracts features
├── generate_negative_pairs.py         # Helper: Creates negative dataset
├── plot_final_comparison.py           # Generate final comparison plot
│
├── models
│   ├── benchmark.py                       # Model 1: Classical ML (sklearn/xgb)
│   ├── model.py                           # Model 2: Neural Networks (PyTorch)
│   ├── lora_model_protbert.py             # Model 3a: ProtBERT Fine-tuning
│   ├── lora_model_esm2.py                 # Model 3b: ESM-2 Fine-tuning
│
├── Output
│   ├── curated_data/                      # PROCESSED DATA (Folds, Features, Labels)
│   ├── cache/                             # Embeddings cache (prevents re-computing)
│   ├── plot/                              # Performance plots
│   └── *_results.txt                      # Detailed logs
```

## 🔬 Technical Details
*   **Handcrafted Features:** 75-dimensional vector using Segmented Composition (N-term, Middle, C-term) capturing Hydrophobicity, Charge, MW, and Amino Acid counts.
*   **ESM-2:** Uses `facebook/esm2_t33_650M_UR50D` for embeddings (Model 1/2) and `esm2_t30_150M_UR50D` for LoRA (Model 3) to fit in standard GPU memory.
*   **ProtBERT-BFD:** Uses `Rostlab/prot_bert_bfd` for Model 3a, offering superior pre-training on 2B+ protein sequences compared to standard ProtBERT.
*   **Caching:** All heavy computations (embeddings) are cached in `cache/` to allow fast iteration on model architectures.
