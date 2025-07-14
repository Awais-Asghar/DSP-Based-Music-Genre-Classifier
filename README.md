# DSP-Based Music Genre Classifier

![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![Tech Stack](https://img.shields.io/badge/Tech-DSP%20%7C%20ML%20%7C%20CNN-pink.svg)
![Platform](https://img.shields.io/badge/Platform-MATLAB-blue.svg)
![Language](https://img.shields.io/badge/Language-MATLAB-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/119c995f-f813-46a3-9d94-a75932e3ad4f" />

---

## Abstract

This project presents a real-time DSP-based music genre classification system that combines handcrafted feature extraction with deep learning techniques. The system integrates classical machine learning models—K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Boosted Trees—with a lightweight Convolutional Neural Network (CNN) trained on mel-spectrograms. The hybrid model achieves high accuracy and low latency suitable for real-time applications such as music streaming platforms, DJ software, and media organization tools.


## Problem Statement

Classical approaches used features like MFCCs and models such as k-NN or SVM to achieve ~70% accuracy. With the rise of deep learning, CNNs operating on spectrogram images boosted performance beyond 85%. However, real-time constraints limit deep model usage on low-resource platforms. This work proposes a hybrid DSP + CNN solution to achieve both speed and performance. So the goal is to design a music genre classification system that balances:

- High accuracy using deep learning
- Low latency using classical models
- Real-time feasibility for live applications


## System Overview
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/5e41204f-7250-4442-ba08-a12a187aec64" />


### DSP Features Extracted

- **Mel-Frequency Cepstral Coefficients (MFCCs)** – captures timbral textures
- **Spectral Descriptors** – centroid, rolloff, flux, ZCR
- **Rhythmic Features** – beat strength, BPM, tempo variance
- **Harmonic Features** – pitch histograms, chroma vectors

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/ac7f80b0-dfbe-4e3f-8dbd-b0f3c76292f8" />

---

### CNN on Mel-Spectrograms

- **Preprocessing**:
  - Frame audio with 2048-sample windows, 50% overlap
  - Apply FFT, Mel filterbanks, and log compression
  - Generate 64×128 mel-spectrogram image for 3s audio clips

- **CNN Architecture**:
  - Two Conv layers (64 filters, 3×3 and 3×5)
  - Max-pooling (2×4)
  - One dense layer (32 units) + Softmax
  - Dropout: 0.3, Optimizer: Adam (lr = 0.001)


## Performance Comparison

| Model           | Accuracy | Precision | Recall |
|----------------|----------|-----------|--------|
| k-NN           | 60.5%    | 0.61      | 0.60   |
| SVM            | 68.7%    | 0.70      | 0.69   |
| Boosted Trees  | 75.4%    | 0.76      | 0.75   |
| CNN            | 83.0%    | 0.84      | 0.83   |
| **Hybrid Model** | **85.0%** | **0.86** | **0.85** |

> The hybrid model provides the best overall performance with low latency and high generalization.


## Folder Structure

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/1618eefb-985e-4f76-9625-49f6d58f7101" />

---

## How to Run

### Requirements

- MATLAB R2020 or newer  
- Signal Processing Toolbox  
- Deep Learning Toolbox  

### Steps

1. **Clone the repository:**

```bash
git clone https://github.com/Awais-Asghar/DSP-Based-Music-Genre-Classifier
```
2. Open MATLAB and add all folders to path:

```bash   
addpath(genpath('DSP-Based-Music-Genre-Classifier'))
```
3. Run any model:
   
```bash 
run('code/KNN.m')
run('code/SVM.m')
run('code/Boosted_Trees.m')
run('CNN/cnn.m')
```

## Dataset

- Based on **Free Music Archive (FMA)** dataset  
- Genres used: **Folk**, **Hip-Hop**, **Instrumental**, **International**  
- Audio clips are processed into **3-second segments** for classification

---

## Visual Results

Top influential features include:

- **MFCC₂**
- **Spectral Flux**
- **Spectral Kurtosis**
- **Harmonic Ratio**
- **MFCC₃**

These features play a critical role in effectively distinguishing between music genres.
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/65b10599-e4d1-4882-b538-ae8b24900842" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/5a2735ce-d050-4767-a45a-8de788bec7cd" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/90b2f4cd-63f2-40a5-b320-efa36247d02a" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/16ec3fbb-3464-4e66-944a-f80dc70e59b2" />

## Future Work

- Extend to sub-genres and multilingual music
- Add LSTM/GRU layers for temporal modeling
- Optimize CNN using quantization/pruning for embedded systems


## Authors

- **Awais Asghar**
- **Muhammad Ashar Javid**
- **Muhammad Hammad Sarwar**
- **Huzaifa Ahmad**  
  Department of Electrical Engineering,  
  National University of Sciences and Technology (NUST), Islamabad, Pakistan
