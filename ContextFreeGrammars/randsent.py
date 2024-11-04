#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = {}
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file: str):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """
        
        with open(grammar_file, 'r') as f:
            grammar_str = f.read()
            
            
        
        grammar_rules = [line.strip() for line in grammar_str.splitlines() if line.strip() and not line.strip().startswith("#")]

        for rule in grammar_rules:
            prob, lhs, rhs = rule.split("\t")
            # changed int to float as probabilities can also be floats (ex. wallstreet.gr)
            self.rules.setdefault(lhs, []).append((float(prob), rhs.split("#")[0].strip()))           

    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        expansions_left = [max_expansions]

        def recurse(symbol):
            if expansions_left[0] <= 0:
                return '...'

            # if symbol is a terminal, return it
            if symbol not in self.rules:
                return symbol

            # if we get here, symbol is a nonterminal
            expansions_left[0] -= 1
            probs, possible_symbols = zip(*self.rules[symbol])
            
            prob_sum = sum(probs)
            probs = [prob / prob_sum for prob in probs]
        
            next_symbol = random.choices(possible_symbols, probs)[0]
            
            # ensures we retain terminal symbols that are beyond nonterminals (ex. S ., we meed to retain .)
            parts = []
            for sub_symbol in next_symbol.split():
                sub_symbol = sub_symbol.strip()
                if sub_symbol in self.rules:
                    parts.append(recurse(sub_symbol))
                else:
                    parts.append(sub_symbol)
            
            if derivation_tree:
                out = f"({symbol} {' '.join(parts)})"
            else:
                out = ' '.join(parts)
            
            return out

        return recurse(start_symbol)



####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
                derivation_tree=args.tree,
                max_expansions=args.max_expansions,
                start_symbol=args.start_symbol
            )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
