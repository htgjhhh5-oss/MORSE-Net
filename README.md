# MORSE-Net

Official implementation of **MORSE-Net: A Structured Multi-Representation Co-Reasoning Network for Multi-Label 12-Lead ECG Diagnosis**.

The complete implementation code will be organized and made publicly available in this repository.

MORSE-Net is a structured multi-representation co-reasoning framework for multi-label 12-lead ECG diagnosis. The model jointly exploits morphology-aware temporal patterns, rhythm-frequency representations, spatial lead-region information, and diagnostic label dependencies to improve ECG classification performance.

---

## Overview

<p align="center">
  <img src="./img/Figure_1.jpg" alt="Overview of MORSE-Net" width="100%">
</p>

<p align="center">
  <b>Figure 1.</b> Overall architecture of MORSE-Net.
</p>

---

## Requirements

The implementation is based on Python and PyTorch. The main dependencies are listed below:

```text
Python >= 3.8
PyTorch >= 1.10.0
NumPy >= 1.21.0
SciPy
Pandas
Scikit-learn
WFDB
TQDM
Matplotlib
PyYAML
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Configuration

Training and evaluation settings are defined in:

```text
config.py
```

The configuration file includes dataset paths, number of diagnostic classes, training hyperparameters, and output directories.

### Training

After preparing the dataset and setting the configuration file, run:

```bash
python main_train.py
```

For the MiniRocket-based training strategy, run:

```bash
python minirocket_train.py
```

---

## Data Preparation

In the experiments, ECG recordings are resampled to **100 Hz** and standardized to **1000 samples per lead**. Each ECG sample is represented as:

```text
12 × 1000
```

For recordings longer than 10 seconds, the first 1000 samples are used. For recordings shorter than 10 seconds, zero-padding is applied.

---

## Input Format

The expected input tensor format is:

```text
batch_size × 12 × 1000
```

where `12` denotes the number of ECG leads and `1000` denotes the number of time samples after preprocessing.

---

## Datasets

MORSE-Net was evaluated on the following public 12-lead ECG datasets.

| Dataset          | Description                                                         | Link                                                               |
| ---------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| PTB-XL           | A large-scale public 12-lead ECG dataset                            | https://physionet.org/content/ptb-xl/                              |
| CPSC 2018        | A 12-lead ECG dataset from the China Physiological Signal Challenge | http://2018.icbeb.org/Challenge.html                               |
| HFHC             | A multi-label ECG dataset from the Tianchi ECG competition          | https://tianchi.aliyun.com/competition/entrance/231754/information |
| Chapman-Shaoxing | A 12-lead ECG dataset with a large diagnostic label space           | https://physionet.org/content/ecg-arrhythmia/1.0.0/                |

---

## Model Architecture

MORSE-Net consists of three major components.

### Morphology--Rhythm--Spatial Representation Decomposition

This component extracts complementary ECG representations from 12-lead ECG signals, including temporal morphology features, rhythm-frequency features, and morphology-gradient features.

### Cross-Representation Interaction and Residual Aggregation

This component models interactions among different ECG representations and performs adaptive multi-view feature aggregation through residual learning.

### Dual-Stage Diagnostic Label Graph Reasoning

This component captures diagnostic label dependencies and enhances multi-label prediction through structured label-level reasoning.

---

## Project Structure

```text
MORSE-Net/
├── img/
│   └── Figure_1.jpg
├── config.py
├── main_train.py
├── minirocket_train.py
├── requirements.txt
└── README.md
```

