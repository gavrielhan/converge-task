#!/usr/bin/env python3
"""
Final Comparison Plot Script
============================

This script parses the results files from Model 1 (Benchmark) and Model 2 (Neural Networks)
and Model 3 (LoRA) to plot a side-by-side comparison of the best performing models.

For Model 1, it shows both ESM-2 embeddings and Handcrafted features results.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE_1 = RESULTS_DIR / "model1_results.txt"
RESULTS_FILE_2 = RESULTS_DIR / "model2_results.txt"
RESULTS_FILE_3A = RESULTS_DIR / "lora_protbert_predictions.txt"
RESULTS_FILE_3B = RESULTS_DIR / "lora_esm2_predictions.txt"
PLOT_DIR = PROJECT_ROOT / "plot"
PLOT_DIR.mkdir(exist_ok=True)

def parse_model1_results(file_path):
    """Parse Model 1 results to extract best ESM-2 and best Handcrafted scores with std."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return None, 0.0, 0.0, None, 0.0, 0.0

    content = file_path.read_text()
    
    # Find ESM-2 section (new format: "ESM-2 Features:")
    esm_match = re.search(r"ESM-2 Features:(.*?)(?=Handcrafted Features:|$)", content, re.DOTALL)
    esm_section = esm_match.group(1) if esm_match else None
    
    # Find Handcrafted section (new format: "Handcrafted Features:")
    handcraft_match = re.search(r"Handcrafted Features:(.*?)(?=ESM-2 Features:|$)", content, re.DOTALL)
    handcraft_section = handcraft_match.group(1) if handcraft_match else None
    
    # Parse best ESM-2 result (format: "  ClassifierName:\n    ROC-AUC: X ± Y")
    best_esm_score = 0.0
    best_esm_std = 0.0
    best_esm_name = None
    
    if esm_section:
        lines = esm_section.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and line.endswith(':') and not line.startswith('ROC-AUC'):
                classifier_name = line[:-1]
                for j in range(i+1, min(i+10, len(lines))):
                    # Match "ROC-AUC: X ± Y"
                    roc_match = re.search(r'ROC-AUC:\s+([\d\.]+)\s+±\s+([\d\.]+)', lines[j])
                    if roc_match:
                        score = float(roc_match.group(1))
                        std = float(roc_match.group(2))
                        if score > best_esm_score:
                            best_esm_score = score
                            best_esm_std = std
                            best_esm_name = classifier_name
                        break
            i += 1
    
    # Parse best Handcrafted result
    best_handcraft_score = 0.0
    best_handcraft_std = 0.0
    best_handcraft_name = None
    
    if handcraft_section:
        lines = handcraft_section.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and line.endswith(':') and not line.startswith('ROC-AUC'):
                classifier_name = line[:-1]
                for j in range(i+1, min(i+10, len(lines))):
                    roc_match = re.search(r'ROC-AUC:\s+([\d\.]+)\s+±\s+([\d\.]+)', lines[j])
                    if roc_match:
                        score = float(roc_match.group(1))
                        std = float(roc_match.group(2))
                        if score > best_handcraft_score:
                            best_handcraft_score = score
                            best_handcraft_std = std
                            best_handcraft_name = classifier_name
                        break
            i += 1
    
    # Format names
    esm_display = f"Classical ML\n(ESM-2, {best_esm_name})" if best_esm_name else None
    handcraft_display = f"Classical ML\n(Handcrafted, {best_handcraft_name})" if best_handcraft_name else None
    
    return esm_display, best_esm_score, best_esm_std, handcraft_display, best_handcraft_score, best_handcraft_std


def parse_model2_results(file_path):
    """Parse Model 2 results to find the best ROC-AUC score with std."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return None, 0.0, 0.0

    content = file_path.read_text()
    
    best_score = 0.0
    best_std = 0.0
    best_name = None
    
    # Split by Model headers
    blocks = content.split("Model 2")
    for block in blocks[1:]:
        # Extract short name like "Model 2A"
        short_name_match = re.search(r"(Model 2[A-Z])", "Model 2" + block.split("\n")[0])
        # Match "ROC-AUC: X ± Y"
        score_match = re.search(r"ROC-AUC:\s+([\d\.]+)\s+±\s+([\d\.]+)", block)
        
        if short_name_match and score_match:
            name = short_name_match.group(1)
            score = float(score_match.group(1))
            std = float(score_match.group(2))
            if score > best_score:
                best_score = score
                best_std = std
                best_name = f"Neural Net\n({name})"
                
    return best_name, best_score, best_std


def parse_lora_results(file_path):
    """Parse LoRA results to find ROC-AUC score with std (if available)."""
    if not file_path.exists():
        return None, 0.0, 0.0

    content = file_path.read_text()
    # Try to match "ROC-AUC: X ± Y" first, fallback to "ROC-AUC: X"
    score_match = re.search(r"ROC-AUC:\s+([\d\.]+)(?:\s+±\s+([\d\.]+))?", content)
    
    if score_match:
        score = float(score_match.group(1))
        std = float(score_match.group(2)) if score_match.group(2) else 0.0
        if "protbert" in str(file_path).lower():
            name = "LoRA\n(ProtBERT)"
        else:
            name = "LoRA\n(ESM-2)"
        return name, score, std
    
    return None, 0.0, 0.0


def main():
    print("Parsing results...")
    
    # Get Model 1 results (both ESM-2 and Handcrafted) with std
    esm_name, esm_score, esm_std, handcraft_name, handcraft_score, handcraft_std = parse_model1_results(RESULTS_FILE_1)
    if esm_name:
        print(f"Best Model 1 (ESM-2): {esm_name} - {esm_score:.4f} ± {esm_std:.4f}")
    if handcraft_name:
        print(f"Best Model 1 (Handcrafted): {handcraft_name} - {handcraft_score:.4f} ± {handcraft_std:.4f}")
    
    # Get best of Model 2 with std
    name2, score2, std2 = parse_model2_results(RESULTS_FILE_2)
    if name2:
        print(f"Best Model 2: {name2} - {score2:.4f} ± {std2:.4f}")
    
    # Get LoRA results if available
    name3a, score3a, std3a = parse_lora_results(RESULTS_FILE_3A)
    name3b, score3b, std3b = parse_lora_results(RESULTS_FILE_3B)
    
    # Pick best LoRA
    if score3a > score3b:
        name3, score3, std3 = name3a, score3a, std3a
    else:
        name3, score3, std3 = name3b, score3b, std3b
    
    if name3:
        print(f"Best Model 3: {name3} - {score3:.4f} ± {std3:.4f}")
    
    # Prepare Data for Plot
    names = []
    scores = []
    stds = []
    colors = []
    
    # Add ESM-2 result
    if esm_score > 0:
        names.append(esm_name)
        scores.append(esm_score)
        stds.append(esm_std)
        colors.append('#2E86AB')  # Blue
    
    # Add Handcrafted result
    if handcraft_score > 0:
        names.append(handcraft_name)
        scores.append(handcraft_score)
        stds.append(handcraft_std)
        colors.append('#6A9BD1')  # Lighter blue
    
    # Add Model 2 result
    if score2 > 0:
        names.append(name2)
        scores.append(score2)
        stds.append(std2)
        colors.append('#A23B72')  # Purple
        
    # Add Model 3 result
    if score3 > 0:
        names.append(name3)
        scores.append(score3)
        stds.append(std3)
        colors.append('#F18F01')  # Orange
    
    if not names:
        print("No results found to plot.")
        return

    # Plotting with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, yerr=stds, color=colors, edgecolor='black', 
                   alpha=0.8, width=0.6, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add values on top (mean ± std format)
    for bar, score, std in zip(bars, scores, stds):
        height = bar.get_height()
        # Position label above error bar
        label_y = height + std + 0.02
        plt.text(bar.get_x() + bar.get_width()/2., label_y,
                 f'{score:.4f}±{std:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title("Best Model Comparison: ROC-AUC Score (Mean ± Std)", fontsize=14, fontweight='bold')
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.ylim(0, max(1.0, max(s + std for s, std in zip(scores, stds)) + 0.15))
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=15, ha='right')
    
    output_path = PLOT_DIR / "final_model_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nComparison plot saved to: {output_path}")

if __name__ == "__main__":
    main()
