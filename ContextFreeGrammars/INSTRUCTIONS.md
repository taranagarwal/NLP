# Context-Free Grammars: PCFG Toolkit

## Overview

This project implements a comprehensive Probabilistic Context-Free Grammar (PCFG) toolkit for natural language processing, developed as part of Johns Hopkins University's advanced NLP coursework (601.465/665). The system provides both sentence generation from probabilistic grammars and CKY parsing capabilities for syntactic analysis.

## Technologies Used

### Programming Languages
- **Python 3** - Main implementation language for sentence generation
- **Perl** - Parsing algorithms and tree formatting utilities

### Core Libraries & Frameworks
- **argparse** - Command-line argument parsing
- **random** - Probabilistic sampling and weighted random selection
- **os/sys** - System interaction and file operations

### Algorithms & Techniques
- **Probabilistic Context-Free Grammars (PCFGs)** - Weighted grammar rule systems
- **CKY (Cocke-Younger-Kasami) Parsing** - Bottom-up parsing algorithm
- **Viterbi Algorithm** - Finding most probable parse trees
- **Recursive Descent with Memoization** - Efficient grammar expansion
- **Weighted Random Sampling** - Probabilistic rule selection

### Data Formats
- **Tab-separated Grammar Files (.gr)** - Custom grammar specification format
- **Bracket Notation** - Derivation tree representation
- **PDF Documentation** - Assignment specifications

### Development Tools
- **Git** - Version control
- **Unix Shell Scripts** - Automation and pipeline integration

## Prerequisites

- Python 3.x installed
- Perl interpreter (typically pre-installed on Unix systems)
- Unix/Linux environment (macOS or Linux recommended)

## Installation

1. Clone or download the project directory
2. Ensure Python 3 is installed:
   ```bash
   python3 --version
   ```
3. Ensure Perl is available:
   ```bash
   perl --version
   ```
4. Make sure executable permissions are set:
   ```bash
   chmod +x parse prettyprint randsent.py
   ```

## Usage

### Generate Random Sentences

Basic sentence generation:
```bash
python3 randsent.py -g grammar.gr -n 5
```

Generate with derivation trees:
```bash
python3 randsent.py -g grammar.gr -t -n 1
```

Advanced options:
```bash
python3 randsent.py -g grammar.gr -s ROOT -n 10 -M 500 -t
```

### Parse Existing Sentences

```bash
echo "the president ate a sandwich" | ./parse -g grammar.gr
```

### Command Line Options

**randsent.py:**
- `-g, --grammar` - Grammar file path (required)
- `-s, --start_symbol` - Start symbol (default: ROOT)
- `-n, --num_sentences` - Number of sentences to generate (default: 1)
- `-M, --max_expansions` - Maximum nonterminal expansions (default: 450)
- `-t, --tree` - Show derivation trees with pretty formatting

**parse:**
- `-g` - Grammar file path
- Additional parsing options available via `./parse --help`

## Project Structure

```
context-free-grammars/
├── randsent.py              # Main Python sentence generator
├── parse                    # Perl CKY parser executable
├── prettyprint             # Perl tree formatting utility
├── dynaparse               # Alternative parser implementation
├── grammar.gr              # Basic grammar rules
├── grammar2.gr             # Extended grammar with adjectives
├── grammar3.gr             # Complex grammar with multiple constructions
├── grammar4.gr             # Advanced grammar with recursion
├── grammar_ec.gr           # Extra credit grammar extensions
├── extra-grammars/         # Additional grammar files
│   ├── holygrail.gr       # Monty Python vocabulary
│   ├── wallstreet.gr      # Wall Street Journal corpus (10K+ rules)
│   └── readme.txt         # Extra grammars documentation
├── answers.txt             # Assignment solutions and examples
├── hw-grammar.pdf          # Assignment specification
└── README.md              # Basic project information
```

## Configuration

### Grammar File Format

Grammar files use tab-separated format:
```
probability \t nonterminal \t production_rule
```

Example:
```
1.0    ROOT    S .
0.5    S       NP VP
0.3    NP      Det Noun
0.7    VP      Verb NP
```

### Naming Conventions
- **Terminals**: lowercase (e.g., "president", "sandwich")
- **Preterminals**: Capitalized (e.g., "Noun", "Verb")
- **Nonterminals**: ALL-CAPS (e.g., "NP", "VP", "ROOT")

## Data

### Grammar Files
- **grammar.gr** - Basic educational grammar (~50 rules)
- **grammar2.gr** - Extended with adjectives and modifiers
- **grammar3.gr** - Complex constructions and recursion
- **grammar4.gr** - Advanced linguistic phenomena
- **grammar_ec.gr** - Extra credit extensions

### Large Datasets
- **wallstreet.gr** - 10,668 rules from Wall Street Journal corpus
- **holygrail.gr** - Monty Python and the Holy Grail vocabulary

### Output Examples
The `answers.txt` file contains example outputs showing:
- Random sentence generation
- Derivation tree structures
- Parsing results with limited expansions

## Key Features

### Probabilistic Sentence Generation
- Weighted random sampling using `random.choices()`
- Probability normalization for consistent rule selection
- Configurable maximum expansion limits to prevent infinite recursion
- Support for both flat sentence output and hierarchical tree structures

### CKY Parsing Algorithm
- Bottom-up chart parsing implementation
- Viterbi algorithm for finding most probable parses
- Support for binary and unary grammar rules
- Cross-entropy calculation for parse probability assessment

### Tree Visualization
- Bracket notation for derivation trees: `(NP (Det the) (Noun president))`
- Pretty-printing with proper indentation
- Integration with Perl formatting utilities

### Grammar Flexibility
- Supports various grammar complexities from simple to Wall Street Journal scale
- Handles both educational and real-world linguistic phenomena
- Custom start symbols and configurable rule probabilities

## Results/Performance

### Sentence Generation
- Generates syntactically valid sentences following probabilistic distributions
- Handles complex recursive structures with cycle prevention
- Supports grammars ranging from 50 to 10,000+ rules efficiently

### Parsing Accuracy
- CKY algorithm provides optimal parsing for context-free grammars
- Viterbi implementation finds most probable parse trees
- Handles ambiguous sentences with multiple valid parses

### Scalability
- Efficient memory usage with recursive implementation
- Configurable limits prevent exponential expansion
- Successfully processes both simple educational and complex real-world grammars

## Troubleshooting

### Common Issues

**"Permission denied" errors:**
```bash
chmod +x randsent.py parse prettyprint
```

**Python path issues:**
```bash
# Use explicit Python 3
python3 randsent.py -g grammar.gr
```

**Grammar file format errors:**
- Ensure proper tab separation (not spaces)
- Check probability values are numeric
- Verify grammar symbols are consistently defined

**Infinite recursion:**
- Increase `--max_expansions` value
- Check grammar rules for proper termination conditions
- Ensure terminal symbols are defined

### Performance Tips
- Use smaller grammar files for testing
- Limit sentence count (`-n`) for large grammars
- Consider `--max_expansions` for complex recursive grammars

## Academic Context

This project was developed for Johns Hopkins University's Natural Language Processing course (601.465/665), focusing on:
- Formal language theory and context-free grammars
- Probabilistic modeling in computational linguistics
- Parsing algorithms and syntactic analysis
- Implementation of core NLP algorithms from scratch

The assignment demonstrates understanding of fundamental NLP concepts while providing practical experience with grammar-based language generation and parsing systems.