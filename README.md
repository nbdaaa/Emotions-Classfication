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
├── requirements.txt
└── README.md
```

## 💻 Implemented Models

The project implements several models with increasing complexity:

1. **Traditional Machine Learning**:
   - **Linear Support Vector Classification (LinearSVC)**: A discriminative classifier that finds the hyperplane that best separates emotions with maximum margin.
   - **Logistic Regression**: A probabilistic model that uses a logistic function to model the probability of an emotion class.
   - **Random Forest**: An ensemble learning method that constructs multiple decision trees and outputs the mode of the classes.
   - **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' theorem with naive independence assumptions between features.
   - **Decision Tree**: A tree-like model that predicts emotions by learning simple decision rules from the data.

2. **Neural Network Models**:
   - **Transformer Encoder**: A self-attention based architecture that captures contextual relationships in text without recurrence.
   - **Transformer with ROPE (Rotary Position Embedding)**: Enhanced transformer model that uses rotary position embeddings for improved representation of sequential information.

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
   - For traditional ML models: `Model/MLModel/MachineLearning.ipynb`
   - For transformer models: See notebooks in Model/ directory

3. **Making Predictions**:
   - Load trained models from `Saved trained model/`
   - Use the prediction functions defined in the respective notebooks

## 📋 Results

Performance metrics for different models on the test set:

### Traditional Machine Learning Models

| Model | Accuracy | Description |
|-------|----------|-------------|
| LinearSVC | 82.21% | Achieves highest accuracy among traditional models with effective linear separation |
| Logistic Regression | 80.43% | Strong performance with probabilistic output and good interpretability |
| Random Forest | 74.70% | Robust ensemble method that handles non-linear relationships |
| Multinomial Naive Bayes | 72.53% | Efficient model that works well with text classification tasks |
| Decision Tree | 70.55% | Simple, interpretable model with decent performance |

### Deep Learning Models

| Model | Accuracy | Training Time | Description |
|-------|----------|---------------|-------------|
| Transformer with ROPE | ~85% | Slow | State-of-the-art model with superior contextual understanding |
| Transformer Encoder | ~78% | Slow | Powerful self-attention mechanism for capturing text relationships |

## 🔮 Future Work

- [ ] Implement additional models (BERT, RoBERTa, etc.)
   - [ ] Simple RNN, GRU, LSTM 
   - [ ] Pretrained model BERT, RoBERTa, DeBERTa
- [ ] Add cross-validation for hyperparameter tuning
- [ ] Create a web demo for real-time emotion prediction with optional model choice

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - ROPE paper


## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the repository owner.
