# Legacy SIMBA Code

**Date Moved:** December 9, 2024  
**Reason:** Code with 0% usage in README workflows

---

## ⚠️ Warning

This code is **NOT** actively maintained and may be outdated.

These modules were identified as having **0% usage** in production workflows:
- Not imported in README scripts (`preprocessing_scripts/`, `training_scripts/`, `inference_scripts/`)
- Not imported in README notebooks (`notebooks/final_tutorials/`)
- Not transitively used by any production module

---

## 📦 Contents

### Comparison & Benchmarking
- `ms2deepscore_comparison.py` - MS2DeepScore benchmark
- `spec2vec_comparison.py` - Spec2Vec benchmark
- `deterministic_similarity.py` - Old similarity methods
- `modified_cosine.py` - Modified cosine baseline

### Plotting Utilities
- `plot.py` - Legacy plotting functions
- `plot_losses.py` - Training loss visualization
- `simba/plotting.py` - Notebook plotting utilities

### ML Experiments
- `ml_model.py` - Alternative ML approach
- `transformers/sklearn_model.py` - scikit-learn baseline

### MALDI Pretraining
- `pretraining_maldi/` - MALDI-specific modules (experimental)

### Ordinal Classification Variants
- `ordinal_classification/embedder_ordinal.py` - Ordinal regression variant
- `ordinal_classification/embedder_multitask_pretrain.py` - Pretraining variant

### Other
- `adducts_cleaning.py` - Adduct preprocessing
- `similarity.py` - Generic similarity interface
- Various utility modules

---

## 🔄 Restoration

If you need any of this code:

```bash
# Copy specific file back to simba/
cp legacy_scripts/simba/plot.py simba/

# Or restore from git history
git log --all --full-history -- "simba/plot.py"
```

---

## 📊 Statistics

- **Total legacy modules:** 35
- **Production modules:** 50
- **Moved on:** 2024-12-09
- **Analysis:** See `simba/PRODUCTION_API.md`
