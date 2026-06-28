# MORSE-Net: A Structured Multi-Representation Co-Reasoning Network for Multi-Label 12-Lead ECG Diagnosis

This is the official implementation of the paper **"MORSE-Net: A Structured Multi-Representation Co-Reasoning Network for Multi-Label 12-Lead ECG Diagnosis"**.

MORSE-Net is a structured multi-representation co-reasoning framework for multi-label 12-lead ECG diagnosis. It decomposes ECG signals into lead-region temporal, spectral rhythm, and morphology-gradient representations, performs cross-representation interaction before classification, and incorporates diagnostic label dependencies through dual-stage label graph reasoning.

## Dependency

The code is implemented in Python and PyTorch. The main dependencies are:

* python >= 3.8
* pytorch >= 1.10.0
* torchvision
* numpy
* scipy
* pandas
* scikit-learn
* wfdb
* tqdm
* matplotlib

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

The training and evaluation settings are defined in the configuration files under `configs/`.
Each configuration file specifies the dataset path, task type, model settings, training parameters, and output directory.

Example configuration files:

```text
configs/ptbxl_all.yaml
configs/ptbxl_form.yaml
configs/ptbxl_rhythm.yaml
configs/cpsc.yaml
configs/hfhc.yaml
configs/chapman_shaoxing.yaml
```

### Stage 1: Data preprocessing

Before training, please prepare the ECG datasets and organize them according to the paths specified in the configuration files.

To preprocess the datasets, run:

```bash
python preprocess.py --config configs/ptbxl_all.yaml
```

All ECG recordings are resampled to 100 Hz and converted into fixed-length tensors with the shape of `12 × 1000`. For recordings longer than 10 seconds, the first 1000 samples are used. For shorter recordings, zero-padding is applied.

### Stage 2: Training

To train MORSE-Net on a specific dataset and task, run:

```bash
python main_train.py --config configs/ptbxl_all.yaml
```

For other tasks, replace the configuration file accordingly:

```bash
python main_train.py --config configs/ptbxl_form.yaml
python main_train.py --config configs/ptbxl_rhythm.yaml
python main_train.py --config configs/cpsc.yaml
python main_train.py --config configs/hfhc.yaml
python main_train.py --config configs/chapman_shaoxing.yaml
```

The trained checkpoints will be saved in the output directory specified in the configuration file.

### Stage 3: Evaluation

To evaluate a trained model, run:

```bash
python main_eval.py --config configs/ptbxl_all.yaml --checkpoint path/to/checkpoint.pth
```

The evaluation script reports Macro-AUC and sensitivity for multi-label ECG diagnosis.

## Dataset

MORSE-Net is evaluated on four public ECG datasets and six multi-label diagnostic tasks.

### PTB-XL

PTB-XL is a large publicly available 12-lead ECG dataset containing 21,837 ten-second recordings.
We use the official patient-wise split, with folds 1--8 for training, fold 9 for validation, and fold 10 for testing.

Evaluated tasks:

* PTB-XL all
* PTB-XL form
* PTB-XL rhythm

Dataset link:
https://physionet.org/content/ptb-xl/

### CPSC 2018

The CPSC 2018 dataset contains 6,877 12-lead ECG recordings with nine diagnostic labels.
Recordings are resampled to 100 Hz and adjusted to 1000 samples per lead.

Dataset link:
http://2018.icbeb.org/Challenge.html

### HFHC

The HFHC dataset contains multi-label ECG recordings with 34 diagnostic labels.
For recordings with eight measured leads, the remaining four limb leads are reconstructed according to standard lead relationships.

Dataset link:
https://tianchi.aliyun.com/competition/entrance/231754/information

### Chapman-Shaoxing

The Chapman-Shaoxing dataset contains 45,152 ten-second 12-lead ECG recordings.
We map diagnostic codes to 94 target labels and use a reproducible ten-fold split.

Dataset link:
https://figshare.com/collections/ChapmanECG/4560497

## Model

MORSE-Net consists of three main components:

1. **Morphology--Rhythm--Spatial Representation Decomposition (MRS-RD)**
   This module extracts lead-region temporal representations, spectral rhythm representations, and morphology-gradient representations from 12-lead ECG signals.

2. **Cross-Representation Interaction and Residual Aggregation (CIRA)**
   This module performs bidirectional interaction between time-domain-related views and time-frequency rhythm views, followed by adaptive view aggregation and residual view-relation modeling.

3. **Dual-Stage Diagnostic Label Graph Reasoning (DDGR)**
   This module models diagnostic label dependencies at both the feature level and the logit level for multi-label prediction.

## Main Results

MORSE-Net is evaluated using Macro-AUC and sensitivity. The main results are summarized below.

| Dataset / Task   | Macro-AUC | Sensitivity |
| ---------------- | --------: | ----------: |
| PTB-XL all       |     94.17 |       71.34 |
| PTB-XL form      |     89.84 |       52.49 |
| PTB-XL rhythm    |     96.57 |       89.32 |
| CPSC             |     95.76 |       77.22 |
| HFHC             |     95.77 |       91.32 |
| Chapman-Shaoxing |     96.81 |       79.01 |

All values are reported in percentages (%).

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{morsenet2026,
  title={MORSE-Net: A Structured Multi-Representation Co-Reasoning Network for Multi-Label 12-Lead ECG Diagnosis},
  author={Wang, Cheng and Jiang, Xiaogao and Chen, Zhencong and Liu, Xu and Wang, Ran and Rui, Xue and Li, Wanggen and Lin, Zongwu},
  journal={Biomedical Signal Processing and Control},
  year={2026}
}
```

## Contact

For questions about the paper or code, please contact:

```text
Wanggen Li
School of Computer and Information, Anhui Normal University
Email: xchen@ahnu.edu.cn

Zongwu Lin
Department of Thoracic Surgery, Zhongshan Hospital, Fudan University
Email: lin.zongwu@zs-hospital.sh.cn
```
