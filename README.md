# Emotions-Classification

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

A deep learning project for classifying emotions (sentiment) in text using various machine learning and state-of-the-art models. This repository implements multiple approaches to sentiment analysis, from traditional machine learning to transformer-based models.

## 🎯 Project Overview

This project aims to classify text into six emotion categories:
- Anger
- Fear
- Joy
- Love
- Sadness
- Surprise

The implementation includes preprocessing pipeline, model training, evaluation, and inference capabilities.

## 📊 Dataset

The dataset consists of textual content labeled with one of six emotion categories. The data is split into:
- Training set (161,613 samples)
- Validation set (53,871 samples)
- Test set (53,871 samples)

Raw data is stored in `Dataset/Raw dataset/emotions.txt` and processed versions are available in `Dataset/Processed dataset/`.

## 🧬 Project Structure

```
Emotions-Classification/
├── Dataset/
│   ├── Raw dataset/
│   └── Processed dataset/
├── Model/
│   ├── MLModel/
│   ├── Transformer/
│   └── TransformerWithROPE/
├── Saved trained model/
├── Data Preprocessing.ipynb
├── softmax regression.ipynb
├── transformer-encoder-classification.ipynb
├── requirements.txt
└── README.md
```

## 💻 Implemented Models

The project implements several models with increasing complexity:

1. **Softmax Regression**: Basic classification model using GPU acceleration
2. **Decision Tree**: Traditional machine learning model for classification
3. **Transformer Encoder**: Advanced deep learning model based on the transformer architecture
4. **Transformer with ROPE (Rotary Position Embedding)**: Enhanced transformer model with improved position encoding

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Emotions-Classification.git
cd Emotions-Classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. **Data Preprocessing**:
   - Run `Data Preprocessing.ipynb` to process raw data

2. **Training Models**:
   - For softmax regression: `softmax regression.ipynb`
   - For transformer models: See notebooks in Model/ directory

3. **Making Predictions**:
   - Load trained models from `Saved trained model/`
   - Use the prediction functions defined in the respective notebooks

## 📋 Results

Performance metrics for different models on the test set:

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Softmax Regression | ~70% | ~0.68 | Fast |
| Decision Tree | ~65% | ~0.62 | Very Fast |
| Transformer Encoder | ~78% | ~0.76 | Slow |
| Transformer with ROPE | ~82% | ~0.80 | Slow |

## 🔮 Future Work

- [ ] Implement additional models (BERT, RoBERTa, etc.)
   - [ ] Simple RNN, GRU, LSTM 
   - [ ] Pretrained model BERT, RoBERTa, DeBERTa
-[ ] Add cross-validation for hyperparameter tuning
-[ ] Create a web demo for real-time emotion prediction with optional model choice

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - ROPE paper


## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the repository owner.
