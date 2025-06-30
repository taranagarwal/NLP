# Smoothed Language Models Project

## Overview

This Natural Language Processing project implements advanced n-gram language models with multiple smoothing techniques for computational linguistics applications. The project demonstrates both traditional statistical approaches and modern neural methods, applied to real-world tasks including spam detection, language identification, and speech recognition. Built as part of Johns Hopkins University's CS465 NLP course, this project showcases understanding of probabilistic modeling, Bayesian inference, and deep learning optimization.

## Technologies Used

### Programming Languages & Frameworks
- **Python 3.9**: Core implementation language with advanced type annotations
- **PyTorch 2.0**: Neural network framework for log-linear models and tensor operations
- **NumPy 1.19**: Numerical computing and matrix operations
- **SciPy 1.5**: Scientific computing utilities

### Machine Learning Libraries
- **torch.nn**: Neural network modules for embedding layers and linear transformations
- **torch.optim**: Optimization algorithms including custom convergent SGD
- **jaxtyping**: Advanced tensor shape and type annotations
- **typeguard**: Runtime type checking for tensor operations
- **tqdm**: Progress tracking for model training

### Development Tools
- **Conda**: Environment management and package distribution
- **more-itertools**: Extended iteration utilities for corpus processing
- **rich**: Enhanced terminal output and logging
- **logging**: Comprehensive logging framework for debugging and monitoring

### NLP Techniques & Algorithms
- **N-gram Models**: Trigram, bigram, and unigram statistical language models
- **Smoothing Methods**: Add-lambda smoothing and hierarchical backoff techniques
- **Word Embeddings**: Pre-trained Word2Vec vectors with CBOW architecture
- **Neural Language Models**: Log-linear models with embedding features
- **Bayesian Text Classification**: Probabilistic categorization using prior and likelihood
- **Sequence Generation**: Probabilistic sampling for sentence generation

### Training Methods
- **Stochastic Gradient Descent**: Custom convergent SGD with diminishing learning rates
- **Vectorized Operations**: Efficient tensor computations using PyTorch
- **Cross-Entropy Loss**: Optimization objective for neural language models
- **L2 Regularization**: Weight decay for preventing overfitting
- **Mini-batch Training**: Efficient gradient estimation on large corpora

### Evaluation & Analysis
- **Perplexity Calculation**: Standard language model evaluation metric
- **Cross-Entropy**: Information-theoretic model comparison
- **Text Categorization**: Binary and multi-class classification accuracy
- **Probability Normalization**: Ensuring valid probability distributions
- **Out-of-Vocabulary Handling**: Robust processing of unknown words

### Data Processing
- **Tokenization**: Whitespace-based token extraction with special symbols
- **Vocabulary Building**: Frequency-based vocabulary construction with thresholds
- **Corpus Preprocessing**: Handling of multilingual and domain-specific text
- **Integerization**: Optional memory-efficient token representation
- **Generator Functions**: Memory-efficient corpus iteration

## Prerequisites

- **Conda**: Package and environment management system
- **Python 3.9**: Required for compatibility with all dependencies
- **Hardware**: CPU sufficient for small models; GPU recommended for neural training
- **Memory**: 8GB+ RAM recommended for large vocabulary models
- **Storage**: 2GB+ for datasets and trained models

## Installation

1. **Clone the repository** (if applicable) or download the project files
2. **Create the conda environment**:
   ```bash
   conda env create -f code/nlp-class.yml
   ```
3. **Activate the environment**:
   ```bash
   conda activate nlp-class
   ```
4. **Verify installation**:
   ```bash
   cd code
   ./build_vocab.py --help
   ```

## Usage

### Building Vocabularies
Create vocabulary files from training corpora:
```bash
# Basic vocabulary with frequency threshold
./build_vocab.py ../data/gen_spam/train/{gen,spam} --threshold 3 --output vocab-genspam.txt

# Character-based vocabulary for language identification
./build_vocab.py ../data/english_spanish/train/{en.1K,sp.1K} --threshold 2 --output vocab-enspan.txt
```

### Training Language Models

#### Add-Lambda Smoothing
```bash
# Basic add-lambda model
./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 ../data/gen_spam/train/gen --output gen.model

# Experiment with different lambda values
./train_lm.py vocab-genspam.txt add_lambda --lambda 0.1 ../data/gen_spam/train/spam --output spam-small-lambda.model
```

#### Backoff Models
```bash
# Hierarchical backoff from trigrams to unigrams
./train_lm.py vocab-genspam.txt add_lambda_backoff --lambda 1.0 ../data/gen_spam/train/gen --output gen-backoff.model
```

#### Neural Log-Linear Models
```bash
# Basic log-linear model with embeddings
./train_lm.py vocab-genspam.txt log_linear \
  --lexicon ../lexicons/words-10.txt \
  --l2_regularization 0.1 \
  --epochs 10 \
  ../data/gen_spam/train/gen \
  --output gen-neural.model

# GPU-accelerated training
./train_lm.py vocab-genspam.txt log_linear \
  --lexicon ../lexicons/words-50.txt \
  --l2_regularization 0.01 \
  --epochs 50 \
  --device cuda \
  ../data/gen_spam/train/gen \
  --output gen-neural-large.model
```

### Model Evaluation and Testing

#### File Probability Calculation
```bash
# Evaluate model on development set
./fileprob.py gen.model ../data/gen_spam/dev/gen/*

# Compare perplexity across models
./fileprob.py gen-backoff.model ../data/gen_spam/dev/gen/*
./fileprob.py gen-neural.model ../data/gen_spam/dev/gen/*
```

#### Text Categorization
```bash
# Binary spam classification
./textcat.py gen.model spam.model 0.7 ../data/gen_spam/dev/{gen,spam}/*

# Language identification
./textcat.py english.model spanish.model 0.5 ../data/english_spanish/dev/{english,spanish}/*
```

#### Sentence Generation
```bash
# Generate random sentences
./trigram_randsent.py gen.model 10 --max_length 20

# Control generation length and quantity
./trigram_randsent.py spam.model 5 --max_length 15
```

### Advanced Training Options

#### Training Size Experiments
```bash
# Train on different corpus sizes
./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 ../data/gen_spam/train/gen-times2 --output gen-2x.model
./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 ../data/gen_spam/train/gen-times4 --output gen-4x.model
./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 ../data/gen_spam/train/gen-times8 --output gen-8x.model
```

#### Hyperparameter Tuning
```bash
# Lambda parameter sweep for add-lambda models
for lambda in 0.001 0.01 0.1 1.0 10.0; do
  ./train_lm.py vocab-genspam.txt add_lambda --lambda $lambda ../data/gen_spam/train/gen --output gen-lambda-$lambda.model
done

# L2 regularization sweep for neural models
for l2 in 0.001 0.01 0.1 1.0; do
  ./train_lm.py vocab-genspam.txt log_linear --lexicon ../lexicons/words-20.txt --l2_regularization $l2 --epochs 20 ../data/gen_spam/train/gen --output gen-l2-$l2.model
done
```

## Project Structure

### Core Modules
- **`probs.py`**: Main module containing all language model implementations
  - `LanguageModel`: Abstract base class with common functionality
  - `AddLambdaLanguageModel`: Basic add-lambda smoothing implementation
  - `BackoffAddLambdaLanguageModel`: Hierarchical backoff model
  - `EmbeddingLogLinearLanguageModel`: Neural model with word embeddings
  - `ImprovedLogLinearLanguageModel`: Enhanced neural architecture
- **`train_lm.py`**: Model training script with hyperparameter management
- **`fileprob.py`**: Probability evaluation and perplexity calculation
- **`textcat.py`**: Bayesian text categorization implementation
- **`trigram_randsent.py`**: Probabilistic sentence generation
- **`build_vocab.py`**: Vocabulary construction with frequency filtering

### Supporting Code
- **`integerize.py`**: Memory-efficient token representation
- **`SGD_convergent.py`**: Custom convergent SGD optimizer
- **`nlp-class.yml`**: Conda environment specification

### Data Organization
- **`data/english_spanish/`**: Bilingual government documents for language identification
  - `train/`: Training corpora (en.1K, en.2K, etc. and sp.1K, sp.2K, etc.)
  - `dev/`: Development sets for model validation
  - `test/`: Final evaluation sets
- **`data/gen_spam/`**: Email corpus for spam detection
  - `train/`: Training data in multiple sizes (gen, gen-times2, gen-times4, gen-times8)
  - `dev/`: Development sets for hyperparameter tuning
  - `test/`: Test sets for final evaluation
- **`data/speech/`**: Switchboard corpus for speech recognition
  - `train/`: Human transcriptions for language model training
  - `dev/` and `test/`: Candidate transcriptions for ranking
- **`lexicons/`**: Pre-trained word embeddings
  - `words-*.txt`: Wikipedia-trained embeddings (10, 20, 50, 100, 200 dimensions)
  - `words-gs-*.txt`: Genre-specific embeddings including spam data
  - `chars-*.txt`: Character-level embeddings for language identification

### Model Storage
- Models saved as pickled Python objects with `.model` extension
- Descriptive filenames encoding hyperparameters and training data
- Example: `corpus=gen~vocab=vocab-genspam.txt~smoother=add_lambda~lambda=1.0.model`

## Model Types & Algorithms

### Add-Lambda Smoothing
Traditional smoothing technique adding pseudo-counts to all n-grams:
- **Formula**: P(z|xy) = (c(xyz) + λ) / (c(xy) + λ|V|)
- **Hyperparameter**: λ (lambda) controls smoothing strength
- **Applications**: Baseline models, quick experimentation

### Backoff Models
Hierarchical smoothing backing off from trigrams to bigrams to unigrams:
- **Strategy**: Use higher-order n-grams when available, back off to lower orders
- **Implementation**: Recursive probability calculation with interpolation
- **Advantages**: Better handling of sparse data, principled smoothing

### Log-Linear Models
Neural language models using word embeddings:
- **Architecture**: Linear transformation of concatenated context embeddings
- **Features**: Pre-trained Word2Vec embeddings (CBOW method)
- **Training**: SGD with cross-entropy loss and L2 regularization
- **Optimization**: Custom convergent SGD with diminishing learning rates

### Improved Log-Linear Models
Enhanced neural architectures with additional features:
- **Extensions**: Custom improvements to base log-linear model
- **Flexibility**: Inherits from base class, allows method overriding
- **Research**: Opportunity for architectural experimentation

## Configuration

### Hyperparameter Guidelines
- **Lambda (Add-Lambda)**: Start with 1.0, experiment with 0.1-10.0 range
- **L2 Regularization**: Typical values 0.001-1.0, start with 0.1
- **Epochs**: Neural models typically need 10-50 epochs for convergence
- **Embedding Dimensions**: Balance between 10 (fast) and 200 (expressive)
- **Vocabulary Threshold**: Filter rare words with threshold 2-5

### Training Configuration
- **Device Selection**: Use `--device cuda` for GPU acceleration
- **Batch Processing**: Neural models process trigrams in mini-batches
- **Convergence**: Monitor loss decrease and validation performance
- **Memory Management**: Large vocabularies may require substantial RAM

### Model Selection
- **Baseline**: Start with add-lambda models for rapid prototyping
- **Comparison**: Use identical vocabularies when comparing models
- **Validation**: Evaluate on development sets before final testing
- **Ensemble**: Combine multiple models for improved performance

## Data

### Training Corpora
- **English-Spanish**: 240K+ characters of bilingual government documents
  - Character-level tokenization with accent marks removed
  - Multiple training sizes (1K, 2K, 5K, 10K, 20K, 50K characters)
- **Gen-Spam**: Email messages from real user inboxes (2002-2003)
  - Genuine messages vs. unwanted mass emails
  - Scalable training sets (1x, 2x, 4x, 8x sizes)
- **Switchboard**: Telephone conversations between unacquainted adults
  - 2430 conversations, 240 hours of speech, 2.4M words
  - Human transcriptions for training, machine candidates for testing

### Lexicon Files
- **Format**: First line contains vocabulary size and embedding dimension
- **Content**: Word followed by space-separated float values
- **Coverage**: Words appearing in training data with frequency ≥ 5
- **Special Tokens**: BOS, EOS, OOV, OOL for sequence boundaries and unknown words

### Model Files
- **Format**: Pickled Python objects containing complete model state
- **Contents**: Trained parameters, vocabulary, model type, hyperparameters
- **Loading**: Direct import via pickle, ready for immediate use
- **Portability**: Self-contained files for easy model sharing

## Performance & Results

### Expected Outcomes
- **Perplexity**: Lower values indicate better language models
- **Classification Accuracy**: Typical spam detection accuracy 85-95%
- **Training Time**: Neural models require minutes to hours depending on size
- **Memory Usage**: Vocabulary size and embedding dimensions affect RAM needs

### Evaluation Metrics
- **Cross-Entropy**: Primary optimization objective for neural models
- **Perplexity**: Standard language model evaluation metric (2^cross-entropy)
- **Classification Accuracy**: Percentage of correctly categorized documents
- **F1 Score**: Balanced measure for imbalanced datasets

### Optimization Tips
- **Vectorization**: Use PyTorch tensor operations for speed
- **GPU Acceleration**: Significant speedup for neural model training
- **Vocabulary Size**: Balance between coverage and computational cost
- **Regularization**: Prevent overfitting with appropriate L2 penalties

## Troubleshooting

### Common Issues
- **Out of Memory**: Reduce vocabulary size or embedding dimensions
- **Slow Training**: Enable GPU acceleration or reduce training data size
- **Poor Performance**: Adjust smoothing parameters or increase training data
- **Vocabulary Mismatch**: Ensure consistent vocabulary across compared models

### Environment Setup
- **Conda Conflicts**: Use exact environment specification from `nlp-class.yml`
- **PyTorch Installation**: Verify CUDA compatibility for GPU training
- **Path Issues**: Run scripts from `code/` directory with relative paths
- **Permission Errors**: Ensure write access for model output files

### Model Training
- **Convergence Issues**: Reduce learning rate or increase regularization
- **Gradient Explosion**: Check for numerical instability in loss calculation
- **Validation Performance**: Monitor development set to prevent overfitting
- **Memory Leaks**: Use proper tensor device management in PyTorch

### Data Processing
- **Encoding Issues**: Ensure UTF-8 encoding for international text
- **File Formats**: Verify whitespace-delimited token format
- **Missing Files**: Check data directory structure and file paths
- **Vocabulary Coverage**: Ensure test words appear in training vocabulary

## Advanced Features

### GPU Acceleration
- **Setup**: Use `--device cuda` flag for CUDA-enabled training
- **Requirements**: NVIDIA GPU with CUDA support
- **Memory**: Monitor GPU memory usage for large models
- **Speedup**: 10-100x faster training for neural models

### Custom SGD Implementation
- **Algorithm**: Convergent SGD with diminishing learning rates
- **Theory**: Guaranteed convergence for convex optimization
- **Implementation**: Custom PyTorch optimizer following Bottou (2012)
- **Usage**: Automatic learning rate scheduling based on iteration count

### Type Checking and Validation
- **jaxtyping**: Tensor shape annotations for debugging
- **typeguard**: Runtime type checking for development
- **Documentation**: Self-documenting code through type annotations
- **Debugging**: Early detection of tensor dimension mismatches

### Memory Optimization
- **Integerization**: Convert string tokens to integers for efficiency
- **Generator Functions**: Stream processing of large corpora
- **Lazy Loading**: Load data only as needed during training
- **Garbage Collection**: Proper cleanup of PyTorch tensors and gradients