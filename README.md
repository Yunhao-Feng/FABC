<div align="center">


# 🔬 FABC: Federated Adversarial and Backdoor Defense with Causal Inference

<h4>


*Unified Robustness for Federated Learning under Coexisting Adversarial and Backdoor Threats*



[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square&logo=adobeacrobatreader)](./FABC.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10%2F100%2FSVHN-4B8BBE?style=flat-square)](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📝 Overview

**FABC** is a **unified federated defense framework** that simultaneously protects against **adversarial examples** and **backdoor poisoning attacks** under heterogeneous, non-IID data distributions — without requiring any additional clean data.

Unlike most existing defenses that address only a single threat, FABC jointly handles their **coexistence and interaction** through a principled **causal deconfounding mechanism**. The framework trains two cooperating models: a **proxy model** that captures attack-sensitive spurious correlations, and a **clean model** that learns task-relevant causal representations by minimizing its dependence on the proxy. Only the clean model is aggregated at the server, preventing malicious correlations from propagating into the global model.

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">


### 🧠 Causality-Inspired Design

- Models image classification with a **structural causal graph**
- Separates causal semantic features from **spurious adversarial-backdoor correlations**
- Uses **adversarial mutual-information minimization** between clean and proxy representations via a WGAN-style discriminator

### 🛡️ Dual-Threat Robustness

- **Adversarial robustness**: Best robust accuracy in **10 / 12** attack settings (FGSM, PGD-20, CW, BIM)
- **Backdoor defense**: Reduces average ASR to **0.45%** (86.7% relative reduction over strongest baseline DBD)
- Consistently suppresses BadNets, Trojan, Blend, SIG, and WaNet attacks

### ⚖️ Label-Skew Calibration

- **Calibrated cross-entropy (CCE)** loss adapts to severe label distribution skew
- Absorbs client-specific label priors into logit calibration
- Stabilizes local training and global aggregation under non-IID settings

### 🔗 Unified Framework

- Single unified pipeline replaces separate adversarial + backdoor defenses
- **No extra clean data** required — trains directly on poisoned datasets
- Only the **clean model** is aggregated, isolating malicious signals locally

---

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.png" alt="FABC Architecture Overview" width="90%">
</div>


FABC consists of three core components:

1. **Label-Prior Calibrated Local Learning** — Each client estimates its smoothed label prior $\pi_i$ and trains with a calibrated cross-entropy objective $L_{\text{cce}}$ that adjusts logits before softmax, reducing cross-client model drift under label skew.

2. **Causal Deconfounding Training** — Two models are jointly trained:
   - **Proxy model** $f_P$: early-stopped to capture low-complexity attack shortcuts (adversarial-backdoor features)
   - **Clean model** $f_C$: optimized to learn causal representations decorrelated from $f_P$

3. **Adversarial Discrimination & Weighted CE** — A discriminator distinguishes real $(C, P)$ pairs from shuffled ones, enforcing $\small C \perp P$. An importance-weighting surrogate $w_e(x,y)$ based on relative losses of $f_P$ and $f_C$ approximates causal risk without knowing the latent attack mechanism.

$$
\mathcal{L}_C = \underbrace{\mathbb{E}\big[w_e(x,y) \cdot L_{\text{cce}}(f_C(x), y)\big]}_{\text{Weighted Cross-Entropy}}
\;+\;
\lambda_{\text{adv}}\!\underbrace{\big(\mathbb{E}_{\text{real}}[D(C,P)] - \mathbb{E}_{\text{shuffled}}[D(C,\tilde{P})]\big)}_{\text{Adversarial Dependence Loss}}
$$

---

## 📁 Repository Structure

```
FABC/
├── README.md                # This file
├── FABC.pdf                 # Paper
├── src/
│   ├── client/              # Client-side local training
│   │   ├── local_train.py   # FABC local training loop
│   │   └── calibration.py   # Label-prior calibration (CCE)
│   ├── models/
│   │   ├── wideresnet.py    # WideResNet backbone
│   │   ├── proxy_model.py   # Proxy model f_P
│   │   ├── clean_model.py   # Clean model f_C
│   │   └── discriminator.py # WGAN-style discriminator
│   ├── server/
│   │   └── aggregation.py   # FedAvg server aggregation
│   ├── attacks/
│   │   ├── adversarial.py   # FGSM, PGD, CW, BIM attacks
│   │   └── backdoor.py      # BadNets, Trojan, Blend, SIG, WaNet
│   ├── utils/
│   │   ├── data_utils.py    # Dataset loading & non-IID partition (Dirichlet)
│   │   └── metrics.py       # Evaluation metrics
│   └── main.py              # Entry point
├── configs/                 # Experiment configuration files
├── scripts/                 # Shell scripts for running experiments
└── assets/                  # Figures and illustrations
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU (recommended; tested on 2× RTX 3090)

### Installation

```bash
git clone https://github.com/Yunhao-Feng/FABC.git
cd FABC
pip install -r requirements.txt
```

### Run FABC Training

```bash
# CIFAR-10 with default settings (α=0.1, 150 epochs)
python src/main.py --dataset cifar10 --alpha 0.1 --epochs 150

# CIFAR-100 with custom adversarial attack
python src/main.py --dataset cifar100 --attack pgd --epsilon 8 --epochs 150

# SVHN with specific backdoor attack
python src/main.py --dataset svhn --backdoor badnets --target_label 0
```

---

## 📖 Citation

If you find FABC useful in your research, please consider citing:

```bibtex
@article{chen2025fabc,
  title     = {{FABC}: Federated Adversarial and Backdoor Defense with Causal Inference},
  author    = {Yujin Chen and Yunhao Feng and Yanming Guo and Mingrui Lao},
  journal   = {Knowledge-Based Systems},
  year      = {2025},
  note      = {College of Systems Engineering, National University of Defense Technology},
  url       = {https://github.com/Yunhao-Feng/FABC}
}
```
