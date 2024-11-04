
"""
Samples k sentences from the distribution given by any 
trained model. Will begin each sentence conditioning on `BOS` until reaching
min(max_length, EOS).
"""

import argparse
import logging
from pathlib import Path
import math
import torch

from probs import LanguageModel

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to trained model"
    )
    parser.add_argument(
        "k",
        type=int,
        help="number of sentences to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="maximum length of a given sentence"
    )
    
    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

    

        
        
def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    k = args.k
    lm = LanguageModel.load(args.model)
    max_length = args.max_length
    
    
    sentences: list[str] = []
    
    for i in range(k):
        sentence = lm.sample(max_length)
        print(f"{i+1}: {sentence}")
        sentences.append(sentence)
        
    return sentences
        
    
    
if __name__ == "__main__":
    main()