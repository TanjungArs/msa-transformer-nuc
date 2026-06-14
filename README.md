# 🧬 MSATransformer: Sequence-to-Sequence Deep Learning Model for Nucleotide Multiple Sequence Alignment

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org) 
[![Academic: Thesis](https://img.shields.io/badge/Academic-Skripsi%202026-blue)](http://uinjkt.ac.id)

This repository contains the **research and training implementation** of **MSATransformer**, a deep learning model that reformulates Multiple Sequence Alignment (MSA) as an autoregressive Sequence-to-Sequence (S2S) learning problem.

The system focuses on **model architecture design, dataset preparation, training pipeline, and experimental evaluation**.

Pretrained weights from this repository are used in a separate deployment system.

> 🔄 **Note:** This repository does not contain the inference API or web application. For deployment and real-time usage, refer to the separate repository:  
> 👉 https://github.com/TanjungArs/MSA-Transformer

<br>

## 🔬 Research Motivation

Multiple Sequence Alignment (MSA) is a classical problem in computational biology and is known to be **NP-complete under standard scoring formulations (e.g., Sum-of-Pairs optimization)**. As a result, traditional tools rely on heuristic strategies such as progressive alignment and guide-trees.

These approaches suffer from:
- Error propagation in early alignment stages  
- Dependence on static substitution matrices  
- Limited modeling of long-range dependencies  

<br>

## 🧠 Sequence-as-Language Modeling

This work reframes biological sequences as a **structured language modeling problem**, where:
- Nucleotides are treated as tokens (`A`, `T`, `C`, `G`)
- Gaps represent structural alignment decisions
- Evolutionary dependencies are modeled as contextual relationships

We adopt a **Transformer-based encoder-decoder architecture** to learn alignment patterns directly from data without handcrafted scoring rules.

<br>

## ⚙️ Model Architecture

MSATransformer uses a shared encoder with dual-decoder design:

### 1. Align Decoder
Predicts:
- Aligned nucleotide tokens
- Gap insertion (`-`)

### 2. Gap Decoder
Models:
- Indel boundary structure
- Structural consistency of alignment regions

<br>

### 🏗️ Architecture Overview

```
Input Sequences (FASTA)
            │
            ▼
Tokenization + Concatenation ('|')
            │
            ▼
Sinusoidal Positional Encoding
            │
            ▼
Multi-Head Transformer Encoder
            │
            ▼
Shared Context Representation
            │
    ┌───────────────────────────┐
    ▼                           ▼
Align Decoder              Gap Decoder
│                               │
▼                               ▼
Aligned Tokens            Indel Structure
```

<br>

## 🎯 Training Objective

The model is trained using a multi-task loss:


$$ L_{total} = L_{align} + L_{gap} $$

Where:
- `L_align`: Cross-Entropy loss for nucleotide alignment prediction  
- `L_gap`: Cross-Entropy loss for gap / structural boundary prediction  

Both losses use padding masks (`ignore_index`) to ensure stable gradient computation.

<br>

## 📦 Repository Contents

This repository includes:

- Model architecture (Encoder-Decoder Transformer)
- Tokenization utilities for biological sequences
- Dataset preprocessing pipeline
- Training loop implementation
- Evaluation metrics (SP and CS score)
- Experiment configurations

<br>

## 📊 Evaluation Metrics

### 1. Sum-of-Pairs (SP Score)
Measures pairwise alignment consistency across sequences.

### 2. Column Score (CS Score)
Measures column-wise conservation accuracy across aligned sequences.

<br>

## 📈 Experimental Results

The model is evaluated against classical MSA tools:

- MAFFT  
- Clustal Omega  
- MUSCLE  

Key observations:
- Improved structural consistency in simulated evolutionary datasets  
- Competitive performance in SP and CS metrics  
- Better gap placement consistency compared to heuristic methods  

<br>

## ⚙️ Training Setup

### Requirements
- Python 3.10+
- PyTorch
- NumPy

### Installation
```bash
pip install torch numpy
````

### Training

```bash
python train.py
```

<br>

## 🌐 Related Repository

* 🔵 Deployment / Inference System:
  [https://github.com/TanjungArs/MSA-Transformer](https://github.com/TanjungArs/MSA-Transformer)



## 📌 Notes

* This repository is focused on **training and research**
* Pretrained models are exported as `.pt` checkpoints
* Inference is handled in a separate production system


## 🎓 Research Citation

```bibtex
@thesis{tanjung2026msa,
  author       = {Tanjung Arswendo Yudha},
  title        = {Penerapan Model Sequence-to-Sequence Menggunakan Transformer Untuk Penjajaran Sekuens Nukleotida},
  school       = {Universitas Islam Negeri Syarif Hidayatullah Jakarta},
  year         = {2026},
  type         = {Skripsi},
  department   = {Program Studi Teknik Informatika}
}
```