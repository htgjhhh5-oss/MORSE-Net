# MORSE-Net

Official implementation of **MORSE-Net: A Structured Multi-Representation Co-Reasoning Network for Multi-Label 12-Lead ECG Diagnosis**.
## Figure 1

[View Figure 1](./img/Figure_1.pdf)
MORSE-Net is designed for multi-label 12-lead ECG classification by jointly modeling morphology, rhythm-frequency, spatial lead-region information, and diagnostic label dependencies.

---

## Requirements

The code is implemented in Python and PyTorch. The main dependencies are:

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

Please modify the dataset path, number of classes, training parameters, and output directory according to your own environment.

### Training

After preparing the dataset and setting the configuration file, run:

```bash
python main_train.py
```

If the MiniRocket-based training strategy is used, run:

```bash
python minirocket_train.py
```



## Data Preparation

In our experiments, ECG recordings are resampled to 100 Hz and adjusted to 1000 samples per lead. The input tensor has the shape:

```text
12 × 1000
```

For recordings longer than 10 seconds, the first 1000 samples are used. For shorter recordings, zero-padding is applied.

---

## Datasets

MORSE-Net was evaluated on the following public ECG datasets.

| Dataset          | Description                                                       | Link                                                               |
| ---------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| PTB-XL           | A large public 12-lead ECG dataset                                | https://physionet.org/content/ptb-xl/                              |
| CPSC 2018        | 12-lead ECG dataset from the China Physiological Signal Challenge | http://2018.icbeb.org/Challenge.html                               |
| HFHC             | Multi-label ECG dataset from the Tianchi ECG competition          | https://tianchi.aliyun.com/competition/entrance/231754/information |
| Chapman-Shaoxing | 12-lead ECG dataset with a large diagnostic label space           | [https://figshare.com/collections/ChapmanECG/4560497](https://physionet.org/content/ecg-arrhythmia/1.0.0/)                |

---

## Model

MORSE-Net consists of three main components:

1. **Morphology--Rhythm--Spatial Representation Decomposition**
   Extracts temporal, rhythm-frequency, and morphology-gradient ECG representations.

2. **Cross-Representation Interaction and Residual Aggregation**
   Models interactions among different ECG representations and adaptively aggregates multi-view features.

3. **Dual-Stage Diagnostic Label Graph Reasoning**
   Models diagnostic label dependencies for multi-label ECG prediction.

---

