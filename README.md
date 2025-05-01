# Deep-Learning-based-Language-Translation

This repository presents a comprehensive exploration of neural machine translation (NMT) for English-to-German translation using the WMT14 dataset. It implements and compares three major deep learning architectures:

- *Transformer-based Model*
- *LSTM-based Seq2Seq Model with Attention*
- *Convolutional Sequence-to-Sequence (ConvSeq2Seq) Model*

The project includes modular code, Jupyter notebooks, and pretrained model checkpoints, enabling both in-depth study and practical experimentation.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Model Architectures](#model-architectures)  
3. [Dataset & Preprocessing](#dataset--preprocessing)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
   - [Transformer Model](#transformer-model)  
   - [LSTM Seq2Seq Model](#lstm-seq2seq-model)  
   - [ConvSeq2Seq Model](#convseq2seq-model)  
6. [Results & Evaluation](#results--evaluation)  
7. [References](#references)  
8. [Contact](#contact)  

---

## Project Structure

```

.
├── ConvSeq2Seq-Model/
│   ├── convS2S.py                # Main code for ConvSeq2Seq model
│   ├── convseq2seq-model.pt      # Pretrained ConvSeq2Seq checkpoint
│   └── Readme.md                 # Model-specific instructions
├── LSTM-model/
│   ├── LSTM-model.py             # Main code for LSTM-based model
│   └── README.md                 # Model-specific instructions
├── transformer-model/
│   ├── main.py                   # Training script for Transformer
│   ├── inference.py              # Inference script for Transformer
│   ├── model.py, components.py   # Model architecture and components
│   ├── data.py, utils.py         # Data processing and utilities
│   ├── config.py                 # Configuration and constants
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Model-specific instructions
├── LSTM\_Enc\_Dec.ipynb            # LSTM model notebook
├── NMT\_using\_Transformer.ipynb   # Transformer model notebook
├── convS2S.ipynb                 # ConvSeq2Seq model notebook
└── README.md                     # (You are here)

````

---

## Model Architectures

### 1. Transformer-based Model

- Implements the encoder-decoder architecture with self-attention, positional encoding, and multi-head attention.
- Modularized for easy training, evaluation, and inference.
- Includes utilities for BLEU score calculation and attention visualization.

### 2. LSTM-based Seq2Seq Model with Attention

- 4-layer bidirectional LSTM encoder and 4-layer LSTM decoder.
- Incorporates Luong-style attention for improved alignment and translation quality.
- Serves as a robust baseline for comparison.

### 3. Convolutional Sequence-to-Sequence (ConvSeq2Seq) Model

- Utilizes convolutional layers for both encoder and decoder.
- Employs positional encoding, gated linear units (GLU), residual connections, and dot-product attention.
- Offers an alternative to recurrent and attention-based models.

---

## Dataset & Preprocessing

- **Dataset:** WMT14 English-German parallel corpus.  
- **Preprocessing:** Includes tokenization, lowercasing, and subword segmentation using Byte Pair Encoding (BPE) via SentencePiece.  
- **Data Handling:** Each model directory or notebook contains scripts/functions to download, preprocess, and batch the data as required.

---

## Installation & Setup

### Prerequisites

- Python 3.6 or higher  
- [PyTorch](https://pytorch.org/)  
- [spaCy](https://spacy.io/) (with `de_core_news_sm` and `en_core_web_sm` models)  
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)  
- [Matplotlib](https://matplotlib.org/)  
- [Kaggle API](https://github.com/Kaggle/kaggle-api) (for dataset download)  
- Other dependencies as listed in each model’s `requirements.txt`

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd CSL4020-project

2. **Install dependencies for each model:**

   * **Transformer model:**

     ```bash
     cd transformer-model
     pip install -r requirements.txt
     python -m spacy download de_core_news_sm
     python -m spacy download en_core_web_sm
     ```

   * **ConvSeq2Seq and LSTM models:**

     ```bash
     pip install kaggle pandas spacy torch matplotlib sentencepiece
     ```

3. **Set up Kaggle API (for dataset download):**

   * Place your `kaggle.json` in `~/.kaggle/` and ensure proper permissions.

---

## Usage

### Transformer Model

* **Training:**

  ```bash
  cd transformer-model
  python main.py
  ```

  * Downloads and preprocesses data, builds vocabularies, trains the model, and saves checkpoints.

* **Inference:**

  ```bash
  python inference.py --source_sentence "How are you?"
  ```

  * Loads pretrained weights, translates the input, and can visualize attention.

* **Jupyter Notebook:**

  * Open `NMT_using_Transformer.ipynb` for an interactive demo and training.

### LSTM Seq2Seq Model

* **Jupyter Notebook:**

  * Open `LSTM_Enc_Dec.ipynb` for data preprocessing, training, and evaluation.

* **Script:**

  ```bash
  cd LSTM-model
  python LSTM-model.py
  ```

### ConvSeq2Seq Model

* **Jupyter Notebook:**

  * Open `convS2S.ipynb` for a full pipeline: data download, preprocessing, training, and inference.

* **Script:**

  ```bash
  cd ConvSeq2Seq-Model
  python convS2S.py
  ```

---

## Results & Evaluation

* **Metrics:** BLEU score is used for quantitative evaluation.
* **Visualization:** Attention heatmaps and training curves are available in the notebooks and scripts.
* **Report:** See the `report/` directory for a detailed write-up, including methodology, results, and analysis.

---

## References

* Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
* Gehring et al., ["Convolutional Sequence to Sequence Learning"](https://arxiv.org/abs/1705.03122)
* Luong et al., ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025)
* [WMT14 Dataset](http://www.statmt.org/wmt14/translation-task.html)

---
