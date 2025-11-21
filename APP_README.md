# PPI Prediction Web App 🧬

A simple web application for predicting protein-protein interactions using the trained LightGBM model on ESM-2 embeddings.

## Features

- **Flexible Input**: Enter either:
  - Raw protein sequences (amino acid sequences)
  - Gene names (e.g., TP53, BRCA1) - automatically fetches sequences from UniProt
- **Real-time Prediction**: Uses the trained LightGBM model to predict interactions
- **Confidence Scores**: Shows probability of interaction and no-interaction
- **Beautiful UI**: Modern, responsive interface with gradient backgrounds

## Installation

1. Install required packages:
```bash
pip install -r requirements_app.txt
```

2. Make sure you have trained the LightGBM model:
```bash
python models/benchmark.py
```

This will create checkpoints in `checkpoints/model1/`.

## Running the App

Start the Flask server:
```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

## Usage

### Option 1: Enter Protein Sequences
Paste the amino acid sequences directly into the text boxes:
```
Protein A: MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM
Protein B: MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN
```

### Option 2: Enter Gene Names
Simply type the gene name (the app will fetch the sequence from UniProt):
```
Protein A: TP53
Protein B: MDM2
```

The app automatically detects whether you've entered a sequence or a gene name.

## How It Works

1. **Input Processing**:
   - If input contains only valid amino acids (ACDEFGHIKLMNPQRSTVWY), it's treated as a sequence
   - Otherwise, it's treated as a gene name and the app queries UniProt's REST API

2. **Embedding Generation**:
   - Uses ESM-2 (facebook/esm2_t33_650M_UR50D) to generate embeddings for each protein
   - Mean pooling over sequence length to get fixed-size representations

3. **Feature Engineering**:
   - Creates pair features: `[emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b]`
   - Same feature engineering used during training

4. **Prediction**:
   - Trained LightGBM model predicts interaction probability
   - Returns YES/NO with confidence score

## Example Interactions

### Known Positive Interactions:
- **TP53 & MDM2**: Tumor suppressor and its regulator
- **BRCA1 & BRCA2**: DNA repair proteins
- **EGFR & GRB2**: Growth factor signaling

### Try These:
```
Protein A: TP53
Protein B: MDM2
Expected: YES (high confidence)
```

```
Protein A: BRCA1
Protein B: BRCA2
Expected: YES (moderate to high confidence)
```

## Technical Details

- **Model**: LightGBM trained on protein-disjoint 3-fold CV
- **Embeddings**: ESM-2 (650M parameters)
- **Feature Dimension**: 5120 (1280 * 4 combination methods)
- **Backend**: Flask
- **Frontend**: Vanilla JavaScript with modern CSS

## Performance

- **Model ROC-AUC**: 0.8861 ± 0.0166 (on test set)
- **Embedding Time**: ~1-3 seconds per protein (depends on length and hardware)
- **Prediction Time**: <100ms once embeddings are computed

## Notes

- First prediction may be slow as ESM-2 model loads into memory
- GPU recommended for faster embedding generation (falls back to CPU automatically)
- UniProt queries require internet connection
- Human proteins (organism_id: 9606) are prioritized in gene name searches

## Troubleshooting

**"No LightGBM checkpoints found"**
- Run `python models/benchmark.py` first to train the model

**"transformers not available"**
- Install with: `pip install transformers`

**"Could not find protein sequence for gene: XXX"**
- Check gene name spelling
- Try using the official gene symbol (e.g., TP53 not p53)
- Alternatively, paste the sequence directly

**Slow predictions**
- First prediction loads ESM-2 model (~2GB)
- Subsequent predictions are faster
- Consider using GPU for faster embedding generation

## Future Enhancements

- [ ] Batch prediction for multiple pairs
- [ ] Visualization of attention weights
- [ ] Export results to CSV
- [ ] Support for other organisms
- [ ] Integration with protein databases (STRING, BioGRID)

