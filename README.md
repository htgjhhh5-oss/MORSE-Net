# MORSE-Net

Official PyTorch implementation of **MORSE-Net: A Structured Multi-Representation Co-Reasoning Network for Multi-Label 12-Lead ECG Diagnosis**.



---

## Requirements

The code is implemented in Python and PyTorch.

```text
Python >= 3.8
PyTorch >= 1.10.0
NumPy >= 1.21.0
```

For model-only usage, install:

```bash
pip install torch numpy
```

For integration with a complete ECG training pipeline, the following packages are recommended:

```bash
pip install scipy pandas scikit-learn wfdb tqdm matplotlib pyyaml
```

---

## Repository Structure

```text
models/
├── __init__.py
├── morse_net.py
├── temporal_backbone.py
├── rhythm_frequency_branch.py
├── diagnostic_label_graph.py
└── attention_layers.py
```

This repository provides the MORSE-Net model implementation.

---


In our experiments, ECG recordings are resampled to 100 Hz and adjusted to 1000 samples per lead.

---

## Model Components

MORSE-Net consists of three main components:

1. **Morphology--Rhythm--Spatial Representation Decomposition**
   Extracts temporal, rhythm-frequency, and morphology-gradient ECG representations.

2. **Cross-Representation Interaction and Residual Aggregation**
   Models interactions among different ECG representations and adaptively aggregates multi-view features.

3. **Dual-Stage Diagnostic Label Graph Reasoning**
   Models diagnostic label dependencies at both feature and logit levels for multi-label prediction.

---

## Datasets

MORSE-Net was evaluated on four public ECG datasets:

| Dataset          | Task                      |
| ---------------- | ------------------------- |
| PTB-XL           | all / form / rhythm       |
| CPSC 2018        | multi-label ECG diagnosis |
| HFHC             | multi-label ECG diagnosis |
| Chapman-Shaoxing | multi-label ECG diagnosis |


