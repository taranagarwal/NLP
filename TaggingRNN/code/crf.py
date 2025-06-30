#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Conditional Random Fields.

from __future__ import annotations
import logging
from math import inf, log, exp, isnan
from pathlib import Path
import time
from typing import Callable, Optional, List
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

import itertools, more_itertools
from tqdm import tqdm # type: ignore

from corpus import Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from hmm import HiddenMarkovModel

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomField(HiddenMarkovModel):
    """An implementation of a CRF that has only transition and 
    emission features, just like an HMM."""
    
    # CRF inherits forward-backward and Viterbi methods from the HMM parent class,
    # along with some utility methods.  It overrides and adds other methods.
    # 
    # Really CRF and HMM should inherit from a common parent class, TaggingModel.  
    # We eliminated that to make the assignment easier to navigate.
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize params self.WA and self.WB to small random values, and
        then compute the potential matrices A, B from them.
        As in the parent method, we respect structural zeroes ("Don't guess when you know")."""

        # See the "Training CRFs" section of the reading handout.
        # 
        # For a unigram model, self.WA should just have a single row:
        # that model has fewer parameters.

        self.WA = 0.0001 * torch.rand(self.k if not self.unigram else 1, self.k)
        self.WB = 0.0001 * torch.rand(self.k, self.V)
        self.updateAB()   # compute potential matrices

    def updateAB(self) -> None:
        """Set the transition and emission matrices self.A and self.B, 
        based on the current parameters self.WA and self.WB.
        See the "Parametrization" section of the reading handout."""
       
        # Even when self.WA is just one row (for a unigram model), 
        # you should make a full k × k matrix A of transition potentials,
        # so that the forward-backward code will still work.
        # See init_params() in the parent class for discussion of this point.
        
        self.A = torch.exp(self.WA)
        # self.A[self.bos_t, :], self.A[self.eos_t, :] = 0, 0
        if self.unigram:
            self.A = self.A.repeat(self.k, 1)
        self.B = torch.exp(self.WB)
        # self.B[self.bos_t, :], self.B[self.eos_t, :] = 0, 0
    
    @override
    def train(self,
            corpus: TaggedCorpus,
            loss: Callable[[ConditionalRandomField], float],
            *, 
            tolerance: float = 0.001,
            minibatch_size: int = 1,
            eval_interval: int = 500,
            lr: float = 1.0,
            reg: float = 0.0,
            max_steps: int = 50000,
            save_path: Optional[Path|str] = "my_crf.pkl") -> None:
        """Train the CRF on the given training corpus, starting at the current parameters."""

        def _eval_loss() -> float:
            with torch.no_grad():
                return loss(self)

        if reg < 0: raise ValueError(f"{reg=} but should be >= 0")
        if minibatch_size <= 0: raise ValueError(f"{minibatch_size=} but should be > 0")
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)
        min_steps = len(corpus)

        self._save_time = time.time()
        self._zero_grad()
        steps = 0
        old_loss = _eval_loss()

        for evalbatch in more_itertools.batched(
                            itertools.islice(corpus.draw_sentences_forever(), 
                                                max_steps),
                            eval_interval):
            for sentence in tqdm(evalbatch, total=eval_interval):
                self.accumulate_logprob_gradient(sentence, corpus)
                steps += 1

                if steps % minibatch_size == 0:
                    self.logprob_gradient_step(lr)
                    self.reg_gradient_step(lr, reg, minibatch_size / len(corpus))
                    if save_path: self.save(save_path, checkpoint=steps)
                    self.updateAB()
                    self._zero_grad()

            curr_loss = _eval_loss()
            if steps >= min_steps and curr_loss >= old_loss * (1 - tolerance):
                break
            old_loss = curr_loss

        if save_path: self.save(save_path)

 
    @override
    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Return the *conditional* log-probability log p(tags | words) under the current
        model parameters.  This behaves differently from the parent class, which returns
        log p(tags, words).
        
        Just as for the parent class, if the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  Note that if the sentence is completely
        untagged, then the marginal probability will be 1.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)

        # Remove all tags and re-integerize the sentence.
        # Working with this desupervised version will let you sum over all taggings
        # in order to compute the normalizing constant for this sentence.
        desup_isent = self._integerize_sentence(sentence.desupervise(), corpus)

        log_p_tags_and_words = self.forward_pass(isent)

        log_p_words = self.forward_pass(desup_isent)

        return log_p_tags_and_words - log_p_words


    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Add the gradient of self.logprob(sentence, corpus) into a total minibatch
        gradient that will eventually be used to take a gradient step."""
        
        # In the present class, the parameters are self.WA, self.WB, the gradient
        # is a difference of observed and expected counts, and you'll accumulate
        # the gradient information into self.A_counts and self.B_counts.  
        # 
        # (In the next homework, you'll have fancier parameters and a fancier gradient,
        # so you'll override this and accumulate the gradient using PyTorch's
        # backprop instead.)
        
        # Just as in logprob()
        isent_sup   = self._integerize_sentence(sentence, corpus)
        isent_desup = self._integerize_sentence(sentence.desupervise(), corpus)

        self.E_step(isent_sup, mult=1)

        self.E_step(isent_desup, mult=-1) 
        
    def _zero_grad(self):
        """Reset the gradient accumulator to zero."""
        # You'll have to override this method in the next homework; 
        # see comments in accumulate_logprob_gradient().
        self._zero_counts()

    def logprob_gradient_step(self, lr: float) -> None:
        """Update the parameters using the accumulated logprob gradient.
        lr is the learning rate (stepsize)."""
        
        # Warning: Careful about how to handle the unigram case, where self.WA
        # is only a vector of tag unigram potentials (even though self.A_counts
        # is a still a matrix of tag bigram potentials).
        
        if not self.unigram:
            self.WA += lr * self.A_counts
        else:
            self.WA[0] += lr * self.A_counts.sum(dim=0)  # unigram adjustment
        self.WB += lr * self.B_counts
        self.updateAB() 
        
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        """Update the parameters using the gradient of our regularizer.
        More precisely, this is the gradient of the portion of the regularizer 
        that is associated with a specific minibatch, and frac is the fraction
        of the corpus that fell into this minibatch."""
                    
        # Because this brings the weights closer to 0, it is sometimes called
        # "weight decay".
        
        if reg == 0: return      # can skip this step if we're not regularizing

        decay_factor = 1 - lr * reg * frac
        self.WA *= decay_factor
        self.WB *= decay_factor


        # Warning: Be careful not to do something like w -= 0.1*w,
        # because some of the weights are infinite and inf - inf = nan. 
        # Instead, you want something like w *= 0.9.
 