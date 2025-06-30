# Earley Parser Implementation Project

## Overview

This project implements a probabilistic Earley parser for natural language processing, progressing from a basic context-free grammar recognizer to a highly optimized parsing system. The implementation demonstrates advanced algorithmic design through three distinct phases: `recognize.py` serves as the foundation, implementing Earley's chart-based parsing algorithm for grammaticality checking; `parse.py` extends this to probabilistic parsing using negative log weights to find minimum-weight (highest-probability) parse trees with backpointer chains for tree reconstruction; and `parse2.py` incorporates performance optimizations including vocabulary-based grammar filtering, duplicate prediction prevention, and weight-based pruning. The optimization techniques proved highly effective, achieving a 2.55x speedup on standard English sentences and a remarkable 27.27x improvement on complex Wall Street Journal corpus data by reducing parsing time from over six seconds to 0.24 seconds per sentence. The system showcases practical applications of dynamic programming, priority queue algorithms, and probabilistic modeling in computational linguistics, while effectively handling large-scale grammars through intelligent filtering and pruning strategies that scale with grammar complexity.

## Technologies Used

### Programming Languages & Scripts
- **Python 3** - Main implementation language for parsing algorithms
- **Perl** - Utility scripts for grammar processing and attribute computation
- **Bash** - Shell scripts for workflow automation and testing

### Parsing Algorithms & Data Structures
- **Earley's Algorithm** - Chart-based parsing with PREDICT, SCAN, ATTACH operations
- **Probabilistic Context-Free Grammar (PCFG)** - Weighted grammar rules with probability modeling
- **Chart Parsing** - Dynamic programming approach with agenda-based processing
- **Priority Queues (heapq)** - Best-first parsing with weight-based ordering
- **Backpointer Chains** - Parse tree reconstruction from derivation history
- **Dynamic Programming** - Memoization for efficient parsing

### NLP Techniques & Concepts
- **Context-Free Grammar Formalism** - Syntactic rule representation and processing
- **Negative Log Probability Weights** - Numerical stability in probabilistic calculations
- **Syntactic Parsing** - Parse tree generation and structural analysis
- **Grammar Rule Probability Modeling** - Statistical language modeling
- **Vocabulary Specialization** - Grammar filtering based on input tokens
- **Terminal/Nonterminal Processing** - Symbol classification and rule application

### Optimization Methods
- **Grammar Filtering** - Vocabulary-based rule reduction for large grammars
- **Weight-Based Pruning** - Configurable thresholds for search space reduction
- **Duplicate Prediction Prevention** - Batch processing to eliminate redundant operations
- **Best-First Search** - Priority-based agenda processing for optimal paths
- **Batch Duplicate Detection** - Efficient handling of redundant chart items

### Development Tools
- **Progress Tracking (tqdm)** - Visual progress bars for long parsing operations
- **Python Logging** - Comprehensive debugging and verbosity control
- **Command-Line Interface (argparse)** - Flexible parameter configuration
- **Doctests** - Embedded code validation and documentation
- **Performance Profiling** - Operation counting and timing analysis

### Mathematical Concepts
- **Negative Log Probabilities** - Stable numerical computation of probability products
- **Probabilistic Parsing** - Minimum-weight path finding in parse forests
- **Chart Algorithm Complexity** - Time and space complexity analysis
- **Performance Optimization Scaling** - Algorithmic improvements for large datasets

## Prerequisites

- **Python 3** environment with standard libraries
- **Perl** for utility scripts (buildattrs, checkvocab, etc.)
- Basic understanding of context-free grammars and parsing theory
- Familiarity with command-line interfaces

## Installation

No special installation required. The project uses Python standard libraries:
- `argparse`, `logging`, `math`, `tqdm`, `heapq`
- `dataclasses`, `pathlib`, `collections`, `typing`

## Usage

### Basic Recognition and Parsing

```bash
# Basic grammaticality checking
./recognize.py papa.gr papa.sen
./recognize.py -v papa.gr papa.sen    # verbose output

# Probabilistic parsing with parse tree output
./parse.py papa.gr papa.sen | ./prettyprint
./parse.py --progress english.gr english.sen

# Optimized parsing for large grammars
./parse2.py --progress wallstreet.gr wallstreet.sen
./parse2.py -v english.gr english.sen | ./prettyprint
```

### Grammar Processing and Validation

```bash
# Check vocabulary compatibility
./checkvocab english.gr english.sen

# Build semantic attributes for parse trees
./buildattrs english.gra english.par

# Remove attributes from grammar files
./delattrs english.gra > english.gr
```

### Performance Analysis

```bash
# Progress tracking for large parsing tasks
./parse2.py --progress wallstreet.gr wallstreet.sen

# Detailed operation statistics
./parse2.py -v english.gr english.sen
```

## Project Structure

### Core Parser Implementations
- **`recognize.py`** - Basic Earley recognizer for grammaticality checking
- **`parse.py`** - Probabilistic parser with parse tree reconstruction
- **`parse2.py`** - Optimized version with performance improvements

### Grammar and Data Files
- **`.gr files`** - Context-free grammar rules with probabilities (papa.gr, english.gr, wallstreet.gr)
- **`.gra files`** - Grammar files with semantic attributes
- **`.sen files`** - Input sentences for parsing (papa.sen, english.sen, wallstreet.sen)
- **`.par files`** - Parse tree output in S-expression format

### Utility Scripts
- **`buildattrs`** - Perl script for computing semantic attributes
- **`checkvocab`** - Vocabulary compatibility checker
- **`delattrs`** - Remove attributes from grammar files
- **`prettyprint`** - Format parse tree output
- **`simplify.pl`** - Parse tree simplification utilities

### Example and Test Files
- **`papa.gr/sen`** - Small example grammar and sentences
- **`english.gr/sen`** - Medium-complexity English grammar
- **`wallstreet.gr/sen`** - Large-scale Wall Street Journal grammar

## Algorithm Implementation

### Earley Parsing Components

**Chart-Based Methodology:**
- **Columns**: Represent positions in input sentence (0 to n+1)
- **Items**: Chart entries with rule, dot position, start position, weight, and backpointers
- **Agenda**: Priority queue for processing items in weight-optimal order

**Core Operations:**
- **PREDICT**: Add items for rules expanding expected nonterminals
- **SCAN**: Match terminal symbols against input tokens
- **ATTACH**: Complete items by attaching finished constituents

**Probabilistic Rule Handling:**
- Grammar rules have probability weights converted to negative log base 2
- Item weights accumulate rule probabilities for minimum-weight parsing
- Backpointer chains enable parse tree reconstruction

### Optimization Strategies

**Grammar Filtering (parse2.py):**
- `filter_terminals()` method reduces grammar size based on input vocabulary
- Only keeps rules where all RHS symbols are in input or are nonterminals
- Significantly improves performance on large grammars

**Duplicate Prevention:**
- `predicted_nonterminals` set prevents redundant PREDICT operations
- Batch duplicate checks eliminate repeated agenda additions
- Enhanced item key comparison for efficient duplicate detection

**Weight-Based Pruning:**
- Configurable pruning thresholds focus search on promising paths
- Priority queue processing ensures best-first exploration
- Scales effectively with grammar complexity

## File Formats

### Grammar Files (.gr/.gra)
Tab-delimited format: `<probability>\t<lhs>\t<rhs>`
```
1	ROOT	S
0.8	NP	Det N
0.5	N	caviar
```

### Sentence Files (.sen)
One sentence per line, space-separated tokens:
```
Papa ate the caviar
Papa ate caviar with a spoon
```

### Parse Tree Output (.par)
S-expression format showing hierarchical structure:
```
(ROOT (S (NP Papa) (VP (V ate) (NP (Det the) (N caviar)))))
```

## Performance Optimizations

### Grammar Filtering Techniques
- Vocabulary-based rule reduction for input-specific parsing
- Prevents expansion of irrelevant nonterminals
- Scales linearly with vocabulary size rather than full grammar

### Weight-Based Pruning Strategies
- Configurable thresholds eliminate low-probability paths
- Best-first processing prioritizes promising parses
- Dynamic pruning adapts to parse complexity

### Best-First Parsing Approach
- Priority queues ensure optimal-weight items processed first
- Backpointer management enables efficient tree reconstruction
- Duplicate detection prevents redundant computation

### Scalability for Large Grammars
- 27.27x speedup demonstrated on Wall Street Journal corpus
- Memory-efficient chart representation
- Progressive column processing with tqdm progress tracking

## Data

### Example Grammars
- **papa.gr**: Simple 15-rule grammar for basic testing
- **english.gr**: Medium complexity with hundreds of rules
- **wallstreet.gr**: Large-scale grammar with thousands of rules

### Test Sentences
- **papa.sen**: Simple sentences for initial testing
- **english.sen**: Variety of English constructions
- **wallstreet.sen**: Complex financial news sentences

### Parse Output Structure
- S-expression trees showing syntactic structure
- Probability weights for parse ranking
- Semantic attributes when using .gra grammars

## Testing & Validation

### Doctest Integration
- Embedded tests in class docstrings verify functionality
- Run with `-v` flag for detailed test output
- Agenda class includes comprehensive test examples

### Vocabulary Compatibility Checking
- `checkvocab` script validates sentence/grammar compatibility
- Warns about out-of-vocabulary tokens
- Prevents parsing failures due to missing vocabulary

### Parse Tree Verification
- `prettyprint` utility formats output for readability
- `buildattrs` computes semantic attributes for validation
- Manual inspection of parse structures for correctness

## Troubleshooting

### Common Issues

**Performance Problems:**
- Use `parse2.py` for large grammars
- Enable `--progress` flag for long parsing tasks
- Consider grammar filtering for input-specific parsing

**Parsing Failures:**
- Check vocabulary compatibility with `checkvocab`
- Verify grammar file format (tab-delimited probabilities)
- Ensure sentence tokenization matches grammar terminals

**Memory Issues:**
- Large grammars may require substantial memory
- Consider processing sentences individually
- Monitor progress with verbose output

## Advanced Features

### Semantic Attribute Computation
- `.gra` grammar files support semantic attributes
- `buildattrs` script computes attributes bottom-up
- Lambda calculus integration for compositional semantics

### Lambda Calculus Integration
- `LambdaTerm.pm` Perl module for lambda term manipulation
- Compositional semantic construction during parsing
- Support for complex semantic representations

### Progress Tracking for Large Tasks
- `tqdm` progress bars for column-by-column processing
- Detailed statistics on PREDICT, SCAN, ATTACH operations
- Performance profiling and optimization guidance
