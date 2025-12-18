# SIMBA

![GitHub](https://img.shields.io/github/license/bittremieux-lab/simba)
![Python](https://img.shields.io/badge/python-3.11-blue)

## About SIMBA

SIMBA is a transformer-based neural network that accurately predicts chemical structural similarity from tandem mass spectrometry (MS/MS) spectra. Unlike traditional methods relying on heuristic metrics (e.g., modified cosine similarity), SIMBA directly models structural differences, enabling precise analog identification in metabolomics.

SIMBA predicts two interpretable metrics:

1. **Substructure Edit Distance**: Number of molecular graph edits required to convert one molecule into another.
2. **Maximum Common Edge Substructure (MCES) Distance**: Number of bond modifications required to achieve molecular equivalence.

See the documentation for more information and detailed examples on how to get started with SIMBA for mass spectrometry-based analog discovery.

## Citation

SIMBA is freely available as open source under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

```{toctree}
---
caption: Contents
maxdepth: 1
---

install
quickstart
api
```
