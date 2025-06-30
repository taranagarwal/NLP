#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        if unigram:
            raise NotImplementedError("Unigram CRF is not required for this homework.")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1)  # Dimensionality of word embeddings
        self.E = lexicon

        # Enable fine-tuning of embeddings
        self.E.requires_grad = True

        nn.Module.__init__(self)
        super().__init__(tagset, vocab, unigram)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        self.init_params()


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.
        self.dropout = nn.Dropout(p=0.2)
        
        feature_dim_A = 1 + 2 * self.rnn_dim + 2 * self.k
        feature_dim_B = 1 + 2 * self.rnn_dim + self.k + self.e

        self.M = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.rnn_dim, self.rnn_dim + self.e + 1)))
        self.M_prime = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.rnn_dim, self.rnn_dim + self.e + 1)))

        self.U_a = nn.Sequential(
            nn.Linear(feature_dim_A, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 1)
        )

        self.U_b = nn.Sequential(
            nn.Linear(feature_dim_B, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 1)
        )

        self.theta_a = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.theta_b = nn.Parameter(torch.tensor(0.0, requires_grad=True))
            
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=100, 
            epochs=10
        )
  
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""
        n = len(isent)
        self.H_prefix = torch.empty((n, self.rnn_dim))
        self.H_suffix = torch.empty((n, self.rnn_dim))

        # Prefix vector calculations
        prefix_vectors = []
        prev_h = torch.zeros(self.rnn_dim)
        for j in range(n):
            word_index = isent[j][0]
            col_vector = torch.cat([torch.tensor([1.0]), prev_h, self.E[word_index]], dim=0)
            prev_h = torch.sigmoid(torch.matmul(self.M, col_vector))
            prefix_vectors.append(prev_h)
        self.H_prefix = torch.stack(prefix_vectors)

        # Suffix vector calculations
        suffix_vectors = []
        next_h = torch.zeros(self.rnn_dim)
        for j in reversed(range(n)):
            word_index = isent[j][0]
            col_vector = torch.cat([torch.tensor([1.0]), self.E[word_index], next_h], dim=0)
            next_h = torch.sigmoid(torch.matmul(self.M_prime, col_vector))
            suffix_vectors.append(next_h)
        self.H_suffix = torch.stack(list(reversed(suffix_vectors)))

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        # CODE IF YOU WANT TO CALCULATE AN INDIVIDUAL ELEMENT
        # if position == 1:
        #     prev_prefix_h = torch.zeros(self.rnn_dim)
        # else:
        #     prev_prefix_h = self.H_prefix[position - 2]
        # prev_tag = sentence[position - 1][1]
        # curr_tag = sentence[position][1]
        # curr_suffix_h = self.H.suffix[position]
        # col_vector = torch.cat((1, prev_prefix_h, self.eye[prev_tag], self.eye[curr_tag], curr_suffix_h), 1)
        # feature = torch.nn.Sigmoid(torch.matmul(self.U_a, col_vector))

        if position == 1:
            prev_prefix_h = torch.zeros(self.rnn_dim)
        else:
            prev_prefix_h = self.H_prefix[position - 2]

        curr_suffix_h = self.H_suffix[position]

        tag_pair_first = self.eye.repeat_interleave(self.k, dim=0)
        tag_pair_second = self.eye.repeat(self.k, 1)
        features = torch.cat([
            torch.ones((self.k * self.k, 1)),
            prev_prefix_h.unsqueeze(0).repeat(self.k * self.k, 1),
            tag_pair_first,
            tag_pair_second,
            curr_suffix_h.unsqueeze(0).repeat(self.k * self.k, 1)
        ], dim=1)

        potentials = torch.exp(self.theta_a * self.U_a(features).squeeze(-1))
        return potentials.view(self.k, self.k)
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        word_index = sentence[position][0]
        h_j = self.H_prefix[position]
        h_prime_j = self.H_suffix[position]

        features = torch.cat([
            torch.ones((self.k, 1)),
            h_j.unsqueeze(0).repeat(self.k, 1),
            self.eye,
            self.E[word_index].unsqueeze(0).repeat(self.k, 1),
            h_prime_j.unsqueeze(0).repeat(self.k, 1)
        ], dim=1)

        potentials = torch.exp(self.theta_b * self.U_b(features).squeeze(-1))
        potential_mat = torch.zeros((self.k, self.V))
        potential_mat[torch.arange(self.k), word_index] = potentials
        return potential_mat 