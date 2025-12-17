# 🫀 Cardiac Disease Classification from MRI Sequences

This project implements a **deep learning–based framework for cardiac disease classification using 4D cardiac MRI sequences**.  
The goal is to learn **spatio-temporal patterns across the cardiac cycle** to accurately classify different cardiac conditions.

---

## 🔍 Problem Statement
Cardiac MRI provides rich spatial and temporal information, but effective analysis is challenging due to:
- Variable-length MRI sequences
- Subtle temporal motion differences across cardiac phases
- High-dimensional medical imaging data

This project addresses these challenges using **CNN-based feature extraction combined with temporal modeling architectures**.

---

## 🧠 Methodology

### 1. MRI Data Preprocessing
- Load 4D cardiac MRI volumes (H × W × Z × T)
- Normalize intensity values
- Extract middle-slice temporal sequences
- Resize frames to a fixed resolution
- Handle variable-length sequences using padding

---

### 2. Data Augmentation
To improve generalization, the following augmentations are applied during training:

**Temporal Augmentations**
- Frame dropout
- Speed jitter (temporal resampling)
- Random temporal cropping
- Sequence reversal

**Spatial Augmentations**
- Random cropping
- Small-angle rotations
- Gaussian noise injection

---

### 3. Model Architectures
Multiple architectures are implemented to model temporal dependencies:

- **MobileNet + LSTM**
  - Frame-wise CNN feature extraction
  - Temporal modeling using LSTM

- **MobileNet + ConvLSTM**
  - Preserves spatial structure while modeling temporal dynamics

- **MobileNet + Transformer**
  - Self-attention–based temporal modeling
  - Positional encoding for sequence awareness

All models use **MobileNetV2** adapted for **single-channel MRI input**.

---

## 🏥 Dataset
- **ACDC (Automated Cardiac Diagnosis Challenge) Dataset**
- Cardiac disease classes:
  - NOR – Normal
  - MINF – Myocardial Infarction
  - DCM – Dilated Cardiomyopathy
  - HCM – Hypertrophic Cardiomyopathy
  - RV – Right Ventricle abnormality

---

## 🗂️ Project Structure
```text
├── augmentation.py        # Temporal & spatial data augmentation
├── Data_loader.py         # ACDC MRI dataset loader
├── models.py              # CNN-LSTM, ConvLSTM, Transformer models
├── train.py               # Model training pipeline
├── test.py                # Evaluation and metrics
└── README.md
