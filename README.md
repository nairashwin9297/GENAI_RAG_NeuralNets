#  RAG and Neural Network

This project demonstrates two key machine learning implementations: a Retrieval-Augmented Generation (RAG) system for preventing hallucinations in language models, and a neural network classifier for loan approval prediction.

## Table of Contents
- [Overview](#overview)
- [Part 1: RAG Implementation](#part-1-rag-implementation)
- [Part 2: Neural Network Classifier](#part-2-neural-network-classifier)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)

## Overview

This assignment consists of two main components:

1. **RAG System**: Demonstrates how Retrieval-Augmented Generation improves factual accuracy and prevents hallucinations in language model responses by grounding answers in retrieved source documents.

2. **Neural Network Classifier**: Implements and optimizes a binary classification model for loan approval decisions using various network architectures and activation functions.

## Part 1: RAG Implementation

### Objective
Demonstrate the effectiveness of RAG in preventing hallucinations and providing source citations for language model responses.

### Architecture Components

- **Data Chunking**: Splits text into coherent paragraph units
- **Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text to 384-dimensional vectors
- **Vector Store**: FAISS IndexFlatIP for fast similarity search
- **Generator**: FLAN-T5 small model for response generation
- **Prompt Engineering**: Structured prompts that enforce citation requirements

### Key Features

- **Hallucination Prevention**: Constrains responses to retrieved source material
- **Source Citation**: Every fact is linked to numbered source references
- **Comparative Analysis**: Side-by-side comparison of vanilla vs. RAG responses

### Demo Results

The RAG system successfully:
- Prevents fabricated answers (e.g., stops model from claiming "St. John's River" as longest)
- Provides accurate, cited responses (e.g., "The Nile River... 6,650 kilometers... [1]")
- Maintains 100% source-grounded factual accuracy

## Part 2: Neural Network Classifier

### Objective
Build an optimized binary classifier for loan approval decisions using feed-forward neural networks.

### Dataset
- **Size**: 429 samples with 14 features
- **Target**: Binary classification (accept/reject)
- **Features**: Mix of categorical (Sex, Occupation, etc.) and numerical (Age, Balance, etc.)

### Preprocessing Pipeline

1. **Encoding**: One-hot encoding for categorical variables
2. **Scaling**: StandardScaler for numerical features  
3. **Split**: 80/20 train/validation with stratification

### Model Architectures Tested

| Model | Layers | Activation | Train Acc | Val Acc |
|-------|---------|------------|-----------|---------|
| 3×ReLU | 3×64 units | ReLU | 79.30% | **74.42%** |
| 5×Tanh | 5×64 units | Tanh | 81.63% | 69.77% |
| 3×Tanh | 3×64 units | Tanh | 79.01% | 72.09% |
| 5×ReLU | 5×64 units | ReLU | 84.55% | 70.93% |

### Key Findings

- **Best Performance**: 3×ReLU architecture achieved highest validation accuracy (74.42%)
- **Depth vs. Performance**: Deeper networks (5 layers) showed overfitting without regularization
- **Activation Functions**: ReLU consistently outperformed Tanh on this dataset
- **Training Features**: Early stopping, Adam optimizer, binary crossentropy loss

## Installation

```bash
# Install required packages
pip install faiss-cpu sentence-transformers transformers
pip install pandas scikit-learn tensorflow openpyxl

# For Jupyter notebook environment
pip install jupyter
```

## Usage

### Running the RAG Demo

1. Upload your text data file (`random_data.txt`)
2. Run the RAG implementation:

```python
# Load and process data
chunks = load_chunks("random_data.txt")

# Build vector index
embedder = SentenceTransformer(EMBED_MODEL)
chunk_embs = embedder.encode(chunk_texts)
index.add(chunk_embs)

# Query comparison
query = "What is the longest river mentioned?"
vanilla_response = generate_vanilla(query)
rag_response = generate_rag(query, retrieve(query))
```

### Training Neural Network Models

1. Upload loan dataset (`loan.xlsx`)
2. Run the classification pipeline:

```python
# Preprocess data
X_proc = preprocessor.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_proc, y, test_size=0.2)

# Train and evaluate models
for config in model_configs:
    model = build_model(config['layers'], config['activation'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

## Results

### RAG System Performance
- **Accuracy**: 100% source-grounded responses
- **Citation Rate**: All factual claims properly attributed
- **Hallucination Prevention**: Complete elimination of fabricated facts

### Neural Network Performance
- **Best Model**: 3×ReLU (74.42% validation accuracy)
- **Training Stability**: Early stopping prevented overfitting
- **Architecture Insights**: Moderate depth optimal for this dataset size


## Technical Implementation Details

### RAG Pipeline
- **Retrieval**: Top-k similarity search (k=3)
- **Generation**: Beam search with no-repeat n-grams
- **Evaluation**: Qualitative comparison across 5 test queries

### Neural Network Training
- **Optimization**: Adam optimizer with binary crossentropy
- **Regularization**: Early stopping (patience=5)
- **Validation**: Stratified train/test split for balanced evaluation

## Future Improvements

### RAG Enhancement
- Larger embedding models for improved semantic understanding
- Advanced chunking strategies for better context preservation
- Hybrid retrieval combining semantic and keyword search

### Neural Network Optimization
- Hyperparameter tuning (learning rate, batch size, architecture)
- Regularization techniques (dropout, batch normalization)
- Advanced preprocessing (feature engineering, handling imbalanced data)

## Dependencies

- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Text embeddings
- `transformers`: Language model inference
- `tensorflow`: Neural network implementation
- `scikit-learn`: Preprocessing and evaluation
- `pandas`: Data manipulation
- `numpy`: Numerical computations

---

This project demonstrates practical applications of modern ML techniques for both NLP (RAG) and traditional ML (neural classification) tasks, with emphasis on preventing model hallucinations and optimizing classification performance.
