#!/usr/bin/env python3
"""
Text categorization via Bayes' Theorem
"""
import argparse
import logging
import math
import os
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_1",
        type=Path,
        help="path to the gen trained model",
    )
    parser.add_argument(
        "model_2",
        type=Path,
        help="path to the spam trained model",
    )
    parser.add_argument(
        "prior_prob",
        type=float,
        help="Prior probability that a test file will be of the first category/model"
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob

def get_true_category(file, args):
    # If categories are in the filename, e.g., "gen_file1.txt" or "spam_file2.txt"
    filename = os.path.basename(file)
    if filename.startswith("gen"):
        return args.model_1
    elif filename.startswith("spam"):
        return args.model_2
    else:
        raise ValueError(f"Unable to determine true category for {filename}")




def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    lm_1 = LanguageModel.load(args.model_1, device=args.device)
    lm_2 = LanguageModel.load(args.model_2, device=args.device)
    tot_lm_1, tot_lm_2 = 0, 0

    try:
        assert(lm_1.vocab == lm_2.vocab)
    except:
        raise ValueError("Error: both models must use the same vocabulary")

    total_files = 0
    correct_predictions = 0


    for file in args.test_files:
        log_prob_1: float = file_log_prob(file, lm_1)
        log_prob_2: float = file_log_prob(file, lm_2)
        log_post_1 = log_prob_1 + math.log(args.prior_prob)
        log_post_2 = log_prob_2 + math.log(1 - args.prior_prob)
        filename = os.path.basename(file)
        
        true_category = get_true_category(file, args)

        if log_post_1 > log_post_2:
            predicted_category = args.model_1
            tot_lm_1 += 1
            print(f"{args.model_1} {filename}, {predicted_category == true_category}")
        else:
            predicted_category = args.model_2
            tot_lm_2 += 1
            print(f"{args.model_2} {filename}, {predicted_category == true_category}")


        if predicted_category == true_category:
            correct_predictions += 1
        
        total_files += 1

    print(f"{tot_lm_1} files were more probably from {args.model_1} ({((tot_lm_1/(tot_lm_1+tot_lm_2))*100):0,.2f}%)")
    print(f"{tot_lm_2} files were more probably from {args.model_2} ({((tot_lm_2/(tot_lm_1+tot_lm_2))*100):0,.2f}%)")
    error_rate = 1 - (correct_predictions / total_files)
    print(f"Error rate: {error_rate}")


if __name__ == "__main__":
    main()

