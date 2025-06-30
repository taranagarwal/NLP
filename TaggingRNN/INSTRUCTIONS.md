# Neural CRF Part-of-Speech Tagger

## Overview

This project implements a sophisticated neural sequence labeling system for part-of-speech tagging, progressively building from Hidden Markov Models to state-of-the-art Neural Conditional Random Fields. The implementation demonstrates the evolution from classical probabilistic models to modern deep learning approaches, featuring bidirectional RNNs integrated with non-stationary potential matrices for context-dependent feature extraction.

The project showcases advanced model architecture design with clean inheritance hierarchies, PyTorch integration for automatic differentiation, and comprehensive experimentation frameworks for hyperparameter optimization.

## Technologies Used

### Programming Languages & Frameworks
- **Python 3.9** with advanced type annotations (jaxtyping, typeguard)
- **PyTorch 2.0** for deep learning and automatic differentiation
- **Conda environment management** with PyTorch ecosystem integration

### Deep Learning & Neural Networks
- **Bidirectional Recurrent Neural Networks (biRNN)** for contextual encoding
- **Neural embeddings** and word representations
- **Automatic gradient computation** with backpropagation
- **GPU acceleration support** (CUDA, MPS backends)
- **Non-stationary neural features** for position-dependent modeling

### Machine Learning Models & Algorithms
- **Hidden Markov Models (HMM)** with forward-backward algorithms
- **Conditional Random Fields (CRF)** for sequence labeling
- **Neural CRF** combining probabilistic and neural approaches
- **Viterbi decoding** for optimal sequence prediction
- **Log-space computation** for numerical stability

### NLP Techniques & Methods
- **Part-of-speech tagging** and sequence labeling
- **Non-stationary feature extraction** with position-dependent potentials
- **Word embeddings** and lexicon integration
- **Text integerization** and vocabulary mapping
- **Context-dependent feature functions**

### Optimization & Training
- **Stochastic gradient descent** with PyTorch optimizers
- **L2 regularization** and learning rate scheduling
- **Mini-batch training** with configurable batch sizes
- **Model checkpointing** and training resumption
- **Hyperparameter experimentation** frameworks

### Development Tools
- **Comprehensive evaluation metrics** and cross-entropy computation
- **Progress tracking** with TQDM and logging systems
- **Bash scripting** for experiment automation
- **Jupyter notebooks** for interactive development and testing

## Prerequisites

- **Python 3.9** (conda environment recommended)
- **PyTorch 2.0** with GPU support (optional but recommended)
- **Conda** for environment management
- Required dependencies listed in `nlp-class.yml`

## Installation

1. **Create and activate the conda environment:**
   ```bash
   conda env create -f code/nlp-class.yml
   conda activate nlp-class
   ```

2. **Verify PyTorch installation:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. **Check GPU availability (optional):**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### Basic Model Training

**Hidden Markov Model (baseline):**
```bash
cd code
./tag.py endev --train ensup --model hmm_baseline.pkl
```

**Standard CRF with manual gradients:**
```bash
./tag.py endev --train ensup --crf --model crf_baseline.pkl
```

**CRF with PyTorch backpropagation:**
```bash
./tag.py endev --train ensup --crf --model crf_backprop.pkl
```

**Neural CRF with bidirectional RNN:**
```bash
./tag.py endev --train ensup --crf --rnn_dim 10 --lexicon words-10.txt --model neural_crf.pkl
```

### Advanced Training Configuration

**Full neural model with hyperparameter specification:**
```bash
./tag.py endev --train ensup --crf --reg 0.0001 --lr 0.001 --rnn_dim 10 --batch_size 16 --lexicon words-50.txt --model advanced_model.pkl --max_steps 5000 --eval_interval 500
```

**Training with checkpointing:**
```bash
./tag.py endev --train ensup --crf --rnn_dim 20 --model training_model.pkl --checkpoint existing_model.pkl
```

### Model Evaluation

**Evaluate trained model:**
```bash
./tag.py endev --model trained_model.pkl
```

**Generate predictions with output file:**
```bash
./tag.py testdata --model trained_model.pkl --output predictions.txt
```

### Hyperparameter Experimentation

**Run automated experiments:**
```bash
bash run_experiments.sh
```

**Quick testing with smaller datasets:**
```bash
./tag.py endev --train ensup-tiny --crf --rnn_dim 5 --max_steps 1000
```

## Project Structure

```
nlp-hw7/
├── code/
│   ├── hmm.py                 # Base HMM implementation
│   ├── crf.py                 # Standard CRF with manual gradients
│   ├── crf_backprop.py        # CRF with PyTorch backpropagation
│   ├── crf_neural.py          # Neural CRF with biRNN features
│   ├── crf_test.py            # Non-stationary CRF testing
│   ├── tag.py                 # Main training/evaluation script
│   ├── corpus.py              # Corpus handling and data structures
│   ├── integerize.py          # Text-to-integer mapping utilities
│   ├── lexicon.py             # Word embedding and lexicon management
│   ├── eval.py                # Model evaluation and metrics
│   ├── run_experiments.sh     # Automated experiment runner
│   ├── logsumexp_safe.py      # Numerical stability for log-space
│   ├── nlp-class.yml          # Conda environment specification
│   ├── models/                # Trained model checkpoints
│   ├── evals/                 # Evaluation results
│   └── logs/                  # Training logs
├── data/
│   ├── ensup, endev           # English training/development sets
│   ├── icsup, icdev           # Icelandic datasets
│   ├── possup, posdev         # Position-based test sets
│   ├── nextsup, nextdev       # Next-word test sets
│   └── words-*.txt            # Pre-trained word embeddings
└── INSTRUCTIONS.md            # This file
```

## Model Architecture

### Inheritance Hierarchy
The project implements a clean inheritance structure:
```
HiddenMarkovModel (hmm.py)
└── ConditionalRandomField (crf.py)
    └── ConditionalRandomFieldBackprop (crf_backprop.py)
        └── ConditionalRandomFieldNeural (crf_neural.py)
```

### Hidden Markov Model
- **Forward-backward algorithms** for probability computation
- **Viterbi decoding** for optimal sequence prediction
- **Emission and transition probabilities** with smoothing
- **Log-space computation** for numerical stability

### Conditional Random Field
- **Undirected graphical model** for sequence labeling
- **Feature-based potentials** replacing HMM probabilities
- **Manual gradient computation** using observed vs. expected counts
- **Global normalization** across entire sequences

### CRF with Backpropagation
- **PyTorch nn.Module integration** for automatic differentiation
- **Gradient accumulation** across mini-batches
- **Optimizer integration** (SGD, Adam) for parameter updates
- **Efficient backpropagation** through computation graphs

### Neural CRF with Bidirectional RNN
- **Non-stationary potentials** computed at each position
- **Bidirectional RNN encoding** for left and right context
- **Neural feature functions** combining embeddings and RNN states
- **Position-dependent matrices** A_at(j) and B_at(j)

### Key Neural Architecture Components
- **Word embeddings** E[w] from pre-trained lexicons
- **Tag embeddings** using one-hot representations
- **RNN hidden states** h_j (forward) and h'_j (backward)
- **Feature functions** U_a and U_b for potential computation
- **Scaling parameters** θ_a and θ_b for output dimensions

## Training Configuration

### Hyperparameters
- **Learning rate**: 0.001, 0.0005 (typical values)
- **Regularization**: 0.0001, 0.00005 (L2 penalty)
- **RNN dimensions**: 5, 10, 20, 50 (hidden state size)
- **Batch size**: 1, 16 (mini-batch size)
- **Max steps**: 5000-10000 (training iterations)

### Optimization Settings
- **Optimizer**: SGD with momentum
- **Gradient clipping**: Automatic with PyTorch
- **Learning rate scheduling**: Manual or adaptive
- **Early stopping**: Based on development set performance

### Evaluation Configuration
- **Evaluation interval**: 200-500 steps
- **Metrics**: Cross-entropy (nats), accuracy percentage
- **Checkpointing**: Automatic model saving during training
- **Development set monitoring**: Prevent overfitting

## Performance Optimization

### GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Enable GPU training (automatic detection)
./tag.py endev --train ensup --crf --rnn_dim 20 --model gpu_model.pkl
```

### Efficient Implementation
- **Vectorized tensor operations** in A_at() and B_at()
- **Batch computation** for mini-batch training
- **Lazy vs. eager RNN computation** strategies
- **Memory-efficient gradient accumulation**

### Training Speed Tips
- **Avoid Python loops** in neural computation
- **Use tensor operations** for parallel processing
- **Implement caching** for repeated computations
- **Profile bottlenecks** with PyTorch profiler

## Data

### Training Corpora (do not read large files directly)
- **ensup**: English supervised training (45,000+ sentences)
- **endev**: English development set for evaluation
- **ensup-tiny**: Small subset for quick testing
- **icsup/icdev**: Icelandic datasets for cross-lingual testing

### Special Test Sets
- **possup/posdev**: Position-dependent tagging patterns
- **nextsup/nextdev**: Next-word prediction patterns
- Tests for non-stationary feature effectiveness

### Word Embeddings
- **words-10.txt**: 10-dimensional embeddings (74,981 words)
- **words-50.txt, words-100.txt, words-200.txt**: Higher dimensions
- **One-hot fallback**: Automatic when no lexicon specified

### Data Format
```
# Tagged sentences (word/tag pairs)
The/D cat/N sat/V on/I the/D mat/N ./.
```

## Experimental Results

### Expected Performance Progression
1. **HMM baseline**: ~85-90% accuracy on English POS tagging
2. **Standard CRF**: 1-2% improvement over HMM
3. **Neural CRF**: 2-5% additional improvement with rich features

### Evaluation Metrics
- **Cross-entropy**: Lower is better (measured in nats)
- **Accuracy**: Percentage of correctly tagged tokens
- **Training speed**: Steps per second, convergence time

### Hyperparameter Sensitivity
- **RNN dimension**: Larger dimensions improve accuracy but slow training
- **Learning rate**: 0.001 often optimal, 0.0005 more stable
- **Batch size**: Larger batches more stable but require more memory

## Troubleshooting

### Common Training Issues
- **Gradient explosion**: Reduce learning rate or add gradient clipping
- **Slow convergence**: Increase learning rate or reduce regularization
- **Memory errors**: Reduce batch size or RNN dimensions
- **NaN losses**: Check for log(0) in forward algorithm

### Neural Model Debugging
- **Gradient checking**: Use torch.autograd.gradcheck()
- **Dimension mismatches**: Verify tensor shapes in A_at()/B_at()
- **RNN computation**: Test with small examples first
- **Feature function output**: Check scaling parameters

### Performance Issues
- **Slow training**: Profile with torch.profiler
- **GPU not utilized**: Verify CUDA installation and device placement
- **Memory leaks**: Use torch.no_grad() during evaluation
- **Numerical instability**: Use log-space computation throughout

## Advanced Features

### Non-stationary Potentials
- **Position-dependent features** A_at(j, sentence) and B_at(j, sentence)
- **Context-aware computation** using surrounding words
- **Neural feature extraction** with bidirectional context

### Bidirectional Context Integration
- **Forward RNN**: Left-to-right context encoding
- **Backward RNN**: Right-to-left context encoding
- **Combined representation**: Concatenated hidden states

### Automatic Gradient Computation
- **PyTorch autograd**: Eliminates manual gradient calculation
- **Computation graphs**: Automatic differentiation through neural layers
- **Optimizer integration**: SGD, Adam, other optimizers supported

### Model Checkpointing and Resumption
- **Automatic saving**: Periodic model checkpoints during training
- **Training resumption**: Continue from any checkpoint
- **Best model tracking**: Save model with lowest development loss

### Lexicon and Embedding Management
- **Multiple embedding formats**: Support for various pre-trained embeddings
- **Vocabulary expansion**: Handle out-of-vocabulary words
- **Embedding fine-tuning**: Optional parameter updates during training

This comprehensive implementation demonstrates expertise in both classical probabilistic models and modern neural architectures, showcasing the evolution of NLP techniques and their practical applications in sequence labeling tasks.