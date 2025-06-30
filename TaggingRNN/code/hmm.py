#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###
        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Precompute any quantities needed for forward/backward/Viterbi algorithms.
        This method may be overridden in subclasses."""
        pass
    
    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    @typechecked
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        return self.A
    
    @typechecked
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        return self.B

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # Update emission probabilities (self.B).
        self.B_counts += λ          # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
        self.B[self.eos_t, :] = 0   # replace these nan values with structural zeroes, just as in init_params
        self.B[self.bos_t, :] = 0

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        if not self.unigram:
            assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                    'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
            assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                    'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'
                
        # Update transition probabilities (self.A).  
        # Don't forget to respect the settings self.unigram and λ.
        # See the init_params() method for a discussion of self.A_at in the
        # unigram case.
        self.A_counts += λ
        if self.unigram:
            self.A_counts = self.A_counts.sum(dim=0, keepdim=True).repeat(self.k, 1)
            self.A = torch.divide(self.A_counts, self.A_counts.sum(dim=1, keepdim=True))
        else:
            self.A = torch.divide(self.A_counts, self.A_counts.sum(dim=1, keepdim=True))
            self.A[:, self.bos_t] = 0
            self.A[self.eos_t, :] = 0  

    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              mult: float = 1,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      
      
        dev_loss = loss(self)   # evaluate the model at the start of training
        
        old_dev_loss: float = dev_loss     # loss from the last epoch
        step: int = 0   # total number of sentences the model has been trained on so far      
        while step < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent, mult=mult)

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""
            
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Forward-backward algorithm.
        log_Z_forward = self.forward_pass(isent)
        log_Z_backward = self.backward_pass(isent, mult=mult)
        
        # Check that forward and backward passes found the same total
        # probability of all paths (up to floating-point error).
        #assert torch.isclose(log_Z_forward, log_Z_backward, atol=1e-6), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        self.setup_sentence(isent);
        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign 
        # directly to each alpha[j] in turn.
        n = len(isent) - 2
        alpha = [torch.zeros(self.k) for _ in isent]    
        alpha[0] = self.eye[self.bos_t]  # vector that is one-hot at BOS_TAG
        alpha_log_scal_facts = torch.zeros(n + 2)

        for j in range(1, n + 2):
            word_index = isent[j][0]

            # Get the tensor for previous alpha, with shape (k x 1)
            prev_tag = isent[j - 1][1]
            # If word at j-1 has a tag, then we don't have to consider the other tags' contributions
            # to the alpha calculation.
            prev_alpha = alpha[j - 1] if prev_tag is None else (alpha[j - 1] * self.eye[prev_tag])
            
            # Perform matrix multiplication on alpha(j - 1) and A
            trans_scores = torch.matmul(prev_alpha, self.A_at(j, isent))
            
            # Get p(t | w_i) for all t
            # If we're at EOS, then it's a trivial factor
            emission_prob = self.B_at(j, isent)[:, word_index] if j < n + 1 else (torch.ones(self.k) * self.eye[self.eos_t])

            # Calculate current alpha values, alpha_t_j (j) = (alpha(j - 1) @ A) * B[:, w_j]
            alpha_result = torch.mul(trans_scores, emission_prob)
            scaling_factor = torch.sum(alpha_result)
            # Handle edge case where alpha(j) = 0
            if scaling_factor == 0:
                alpha[j] = torch.tensor(alpha_result)
                scaling_factor = torch.tensor(1)
            # Use scaling trick to avoid underflow
            else:
                alpha[j] = torch.divide(alpha_result, scaling_factor)
            alpha_log_scal_facts[j] = torch.log(scaling_factor)

        self.alpha = alpha
        self.alpha_log_scal_facts = alpha_log_scal_facts 
        log_Z = torch.log(alpha[n + 1][self.eos_t]) + torch.sum(alpha_log_scal_facts)
        self.log_Z = log_Z
        return log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm on a tagged, untagged, or partially tagged sentence.
        Return log Z (the log of the backward probability).
        
        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts."""

        self.setup_sentence(isent);

        n = len(isent) - 2        
        beta = [torch.zeros(self.k) for _ in isent]
        beta[-1] = (self.eye[self.eos_t]) # vector that is one-hot at EOS_TAG
        beta_log_scal_facts = torch.zeros(n + 2)

        for j in range(n + 1, 0, -1):
            word_index = isent[j][0]

            # Get the tensor for beta(j), with shape (k x 1)           
            next_tag = isent[j][1]
            # If word at j has a tag, then we don't have to consider the other tags' contributions
            # to the beta calculation 
            next_beta = beta[j] if next_tag is None else (beta[j] * self.eye[next_tag])

            # Get p(t | w_j) for all t
            # If we're at EOS, then it's a trivial factor
            emission_prob = self.B_at(j, isent)[:, word_index] if j < n + 1 else (torch.ones(self.k) * self.eye[self.eos_t])

            # Calculate preceding beta values beta(j - 1) = A * B[:, w_(j+1)] @ beta(j)
            AB_prod = torch.mul(self.A_at(j, isent), emission_prob)
            beta_result = torch.matmul(AB_prod, next_beta)
            # Use scaling trick to avoid underflow
            scaling_factor = torch.sum(beta_result)
            # Handle edge case where beta(j) = 0
            if scaling_factor == 0:
                beta[j - 1] = torch.tensor(beta_result)
                scaling_factor = torch.tensor(1)
            # Use scaling trick to avoid underflow
            else:
                beta[j - 1] = torch.divide(beta_result, scaling_factor)
            beta_log_scal_facts[j - 1] = torch.log(scaling_factor)

            # If word at j - 1 has a tag, then we don't have to consider the other tags' contributions
            # to the alpha calculation
            curr_tag = isent[j - 1][1]
            curr_alpha = self.alpha[j - 1] if curr_tag is None else (self.alpha[j - 1] * self.eye[curr_tag])

            # Get scaled version of A count updates, and use scaling factors to update A counts 
            # alpha_beta_prod = (torch.divide(torch.matmul(curr_alpha.unsqueeze(0).T, next_beta.unsqueeze(-2)), torch.exp(self.log_Z)))
            alpha_beta_prod = torch.matmul(curr_alpha.unsqueeze(0).T, next_beta.unsqueeze(-2))
            scaled_A_counts = torch.mul(alpha_beta_prod, (AB_prod))
            # Use logspace to get the corrected counts
            log_corr_A_counts = (torch.log(scaled_A_counts) + torch.sum(self.alpha_log_scal_facts[:j]) + torch.sum(beta_log_scal_facts[j:]) - self.log_Z)

            self.A_counts += torch.mul(mult, torch.exp(log_corr_A_counts))
            if word_index == self.vocab.index(EOS_WORD):
                continue

            # Get scaled version of B count updates, and use scaling factors to update B counts 
            # scaled_B_counts = torch.mul(self.alpha[j], next_beta) / exp(self.log_Z)
            scaled_B_counts = torch.mul(self.alpha[j], next_beta)
            # Use logspace to get the corrected counts
            log_corr_B_counts = (torch.log(scaled_B_counts) + torch.sum(self.alpha_log_scal_facts[:j+1]) + torch.sum(beta_log_scal_facts[j:]) - self.log_Z)
            self.B_counts[:, word_index] +=torch.mul(mult, torch.exp(log_corr_B_counts))

        self.beta = beta
        return torch.log(beta[0][self.bos_t]) + torch.sum(beta_log_scal_facts)


    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        isent = self._integerize_sentence(sentence, corpus)
        self.setup_sentence(isent);

        # See comments in log_forward on preallocation of alpha.
        alpha        = [torch.empty(self.k)                  for _ in isent]  
        backpointers = [torch.empty(self.k, dtype=torch.int) for _ in isent]
        tags: List[int]    # you'll put your integerized tagging here

        n = len(isent) - 2
        alpha[0] = self.eye[self.bos_t]
        
        for j in range(1, n + 2):
            word_index = isent[j][0]
            # Get the tensor for previous alpha, with shape (num_tags x 1)
            prev_alpha = alpha[j - 1].view(self.k, 1)
            # prev_tag = isent[j - 1][1]
            # prev_alpha = alpha[j - 1] if prev_tag is None else (alpha[j - 1] * self.eye[prev_tag])

            trans_scores = prev_alpha * self.A_at(j, isent)
            # Get the transition values that maximize the current alpha values
            # max alpha_t_(j-1) (j-1) * p_A(t_j | t_(j-1))
            max_transitions, best_prev_tags = trans_scores.max(dim=0)
            # Get p(t | w_i) for all t
            # If we're at EOS, then it's a trivial factor = 1
            emission_prob = self.B_at(j, isent)[:, word_index] if j < n + 1 else torch.ones(self.k)
            # Calculate current alpha values, alpha_t_j (j)
            alpha_result = max_transitions * emission_prob
            scaling_factor = torch.sum(alpha_result)
            # Handle edge case where alpha(j) = 0
            if scaling_factor == 0:
                alpha[j] = torch.tensor(alpha_result)
            # Use scaling trick to avoid underflow
            else:
                alpha[j] = torch.divide(alpha_result, scaling_factor)
            backpointers[j] = best_prev_tags

        tags = [0] * (n + 2)
        tags[-1] = self.eos_t
        for j in range(n, -1, -1):
            bp_tag_index = tags[j + 1]
            tags[j] = backpointers[j + 1][bp_tag_index].item()

        # Make a new tagged sentence with the old words and the chosen tags
        # (using self.tagset to deintegerize the chosen tags).
        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def save(self, path: Path|str, checkpoint=None, checkpoint_interval: int = 300) -> None:
        """Save this model to the file named by path.  Or if checkpoint is not None, insert its 
        string representation into the filename and save to a temporary checkpoint file (but only 
        do this save if it's been at least checkpoint_interval seconds since the last save).  If 
        the save is successful, then remove the previous checkpoint file, if any."""

        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed

        now = time.time()
        old_save_time =           getattr(self, "_save_time", None)
        old_checkpoint_path =     getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None   # this is a real save, not a checkpoint
        else:    
            if old_save_time is not None and now < old_save_time + checkpoint_interval: 
                return   # we already saved too recently to save another temp version
            path = path.with_name(f"{path.stem}-{checkpoint}{path.suffix}")  # use temp filename
            self._checkpoint_path = path

        # update the elapsed training time since we started training or last saved (if that happened)
        if old_save_time is not None:
            self.total_training_time = old_total_training_time + (now - old_save_time)
        if hasattr(self, "_save_time"):
            del self._save_time
        
        # Save the model with the fields set as above, so that we'll 
        # continue from it correctly when we reload it.
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            # something went wrong with the save; so restore our old fields,
            # so that caller can potentially catch this exception and try again
            self._save_time          = old_save_time
            self._checkpoint_path    = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e
        
        # Since save was successful, remember it and remove old temp version (if any)
        self._save_time = now
        if old_checkpoint_path: 
            try: os.remove(old_checkpoint_path)
            except FileNotFoundError: pass  # don't complain if the user already removed it manually

    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model
 