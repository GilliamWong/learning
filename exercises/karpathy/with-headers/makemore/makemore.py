# makemore.py -- character-level language models over a list of words.
#
# Read a file of words (one per line, e.g. names.txt) and train a model to
# generate new, similar words one character at a time.
#
# Implement: the dataset (words <-> integer index sequences, with start/end
#   tokens), the model zoo -- Bigram, MLP, RNN/GRU, Transformer (and BoW
#   baselines), each mapping character-index sequences to next-character logits
#   and an optional loss -- and autoregressive sampling of new sequences.
# Input:    a words file; CLI flags select the model and hyperparameters.
# Output:   a trained checkpoint and freshly sampled words.
# Behavior: batch -> forward to logits + loss -> backprop -> optimizer step,
#           with periodic evaluation and sampling.


"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    # Resource (GELU behavior): https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    def forward(self, x):
        #input of shape B T C, we want output of same shape, with the tanh approximation
        out = 0.5 * x  * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        return out


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    #self attention that then gets combined with MLP layers to form a Block
    #Blocks are then combined to form the finished transformer model
    #the only thing this self attention class is doing is taking an input of shape (B, T, C), performing self attention on all of the tokens,
    #and then returning

    #things that matter from config:
    # block_size: int = None   # max sequence length
    # n_embd: int = 64
    # n_head: int = 4

    #I think the only thing this class should be doing is attention, we take embeddings as input, perform attention, then softmax to form 
    #attention weights, and then return that. 

    # Resource (Q/K/V projections and multiple heads): https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    def __init__(self, config):
        super().__init__()
        self.len_context = config.block_size
        self.num_heads = config.n_head
        self.embed_dim = config.n_embd
        self.head_dim = self.embed_dim // self.num_heads

        assert config.n_embd % config.n_head == 0

        #need to create Q K and V of the right size, and in pytorch

        self.Q = nn.Linear(self.embed_dim, self.embed_dim)
        self.K = nn.Linear(self.embed_dim, self.embed_dim)
        self.V = nn.Linear(self.embed_dim, self.embed_dim)

        self.merge_heads = nn.Linear(self.embed_dim, self.embed_dim)

        self.register_buffer("attention_mask", ~torch.tril(torch.ones((self.len_context, self.len_context))).bool())


    # Resource (scaled dot-product attention and causal masking): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    def forward(self, x):
        B, T, C = x.size()

        q = self.Q(x) #B T embed_dim
        k = self.K(x) #B T embed_dim
        v = self.V(x) #B T embed_dim

        #need to split into heads, so use .view
        q_heads = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) #these are all B num_heads T head_dim 
        k_heads = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        qk = q_heads @ k_heads.mT # B num_heads T T
        divisor = math.sqrt(self.head_dim) #single value

        raw_scores = torch.div(qk, divisor) #still B num_heads T T 

        masked_scores = raw_scores.masked_fill(self.attention_mask[:T, :T], float('-inf')) #B num_heads T T 

        softmax_scores = torch.softmax(masked_scores, dim=3) #B num_heads T T 
        
        final_scores = softmax_scores @ v_heads #B num_heads T head_dim

        final_scores = final_scores.transpose(1, 2).contiguous() #B T num_heads head_dim
        final_scores = final_scores.view(B, T, self.embed_dim)

        output = self.merge_heads(final_scores) # B T embed_dim
        return output


class Block(nn.Module):
    """ an unassuming Transformer block """

    # Resource (attention + feed-forward block structure): https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
    def __init__(self, config):
        # TODO: implement Block.__init__
        raise NotImplementedError

    # Resource (pre-norm residual block walkthrough): https://www.youtube.com/watch?v=kCc8FmEb1nY
    def forward(self, x):
        # TODO: implement Block.forward
        raise NotImplementedError

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    # Resource (GPT-2 model components and configuration): https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config
    def __init__(self, config):
        # TODO: implement Transformer.__init__
        raise NotImplementedError

    # Resource (n_positions is the maximum context length): https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config
    def get_block_size(self):
        # TODO: implement Transformer.get_block_size
        raise NotImplementedError

    # Resource (causal-LM inputs, logits, labels, and loss): https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.forward
    def forward(self, idx, targets=None):
        # TODO: implement Transformer.forward
        raise NotImplementedError

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, config):
        # TODO: implement CausalBoW.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement CausalBoW.forward
        raise NotImplementedError

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        # TODO: implement BoWBlock.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement BoWBlock.forward
        raise NotImplementedError

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        # TODO: implement BoW.__init__
        raise NotImplementedError

    def get_block_size(self):
        # TODO: implement BoW.get_block_size
        raise NotImplementedError

    def forward(self, idx, targets=None):

        # TODO: implement BoW.forward
        raise NotImplementedError

# -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

#class ModelConfig:
    # block_size: int = None # length of the input sequences of integers
    # vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # # parameters below control the sizes of each model slightly differently
    # n_layer: int = 4
    # n_embd: int = 64 - size of embedding (token) vectors
    # n_embd2: int = 64 - varies by model. For the MLP it's the hidden layer dim/width
    # n_head: int = 4

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        #x_t is of size n_embd and h_t is of size n_embd2. This class has one set of weights for W_hh and then another for W_xh and then concats them
        #the weights should both output n_embd2, so W_hh is (n_embd2 x n_embd2) and W_xh is (n_embd x n_embd2)
        #or we can concat the inputs and just have one linear layer, either or. 
        self.xh_to_h = nn.Linear((config.n_embd + config.n_embd2), config.n_embd2)

    def forward(self, xt, hprev):
        #if this is batched then x_t is B and hprev is just B n_embd2
        #whatever the case we always cat along the last dim to make sure we're fine
        #x embed is passed in as xt
        input = torch.cat((xt, hprev), dim=-1) 
        out = torch.tanh(self.xh_to_h(input))
        return out

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        #we also need an update and a reset gate
        self.update = nn.Linear((config.n_embd + config.n_embd2), config.n_embd2)
        self.reset = nn.Linear((config.n_embd + config.n_embd2), config.n_embd2)

        self.xh_to_h = nn.Linear((config.n_embd + config.n_embd2), config.n_embd2) #same as before


    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        prev_concat = torch.cat((xt, hprev), dim=-1)
        reset_gate = torch.sigmoid(self.reset(prev_concat))
        update_gate = torch.sigmoid(self.update(prev_concat))
        
        #candidate h = (reset gate * prevh) concat with xt, pass into xh_toh
        #new h = update gate * prevh + (1 - update gate) * candidate h

        curr_concat = torch.cat((xt, reset_gate * hprev), dim=-1)
        candidate_h = torch.tanh(self.xh_to_h(curr_concat))

        new_h = (update_gate * hprev) + ((1 - update_gate) * candidate_h)
        return new_h

class RNN(nn.Module):

    def __init__(self, config, cell_type): #so this should just handle the hidden state to y step, I assume
        super().__init__()
        self.block_size = config.block_size
        self.cell_type = cell_type
        self.n_embd = config.n_embd
        self.n_embd2 = config.n_embd2

        self.embeddings = nn.Embedding(config.vocab_size, config.n_embd) 
        self.h = nn.Parameter(torch.zeros(config.n_embd2))
        
        #since we pass off the whole h to h thing to the RNNCell, here we only do the hidden state to output step. So one linear layer. 
        self.h_to_y = nn.Linear(config.n_embd2, config.vocab_size)
        if cell_type == 'rnn':
            self.xh_to_h = RNNCell(config)
        else:
            self.xh_to_h = GRUCell(config)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        #remember that idx is of size B T and targets is of size B T
        if self.cell_type == 'rnn' or self.cell_type == 'gru':
            #in this case T is block_size, but we loop over all tokens in the sequence either way.
            logits = []
            hprev = self.h.expand(idx.shape[0], self.n_embd2) #make the embeds B rows on the first read
            x_embed = self.embeddings(idx) # so now this is B n_embd
            for i in range(idx.shape[1]):
                hprev = self.xh_to_h(x_embed[:, i], hprev) #B x n_embd2
                logits.append(self.h_to_y(hprev)) #will output B x vocab_size

            logits = torch.stack(logits, dim=1) #this is now B T V
            if targets is not None:
                #remember that CE takes B x V size for the logits and B size for the targets, but we have B T V and B T
                #so we flatten such that we have BT V and BT now. 
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1) 
                return logits, loss
            return logits, None
        
        # elif self.cell_type == 'gru':
        #     return None #TODO finish this please

# -----------------------------------------------------------------------------
# MLP language model

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    #class ModelConfig:
    # block_size: int = None # length of the input sequences of integers
    # vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # # parameters below control the sizes of each model slightly differently
    # n_layer: int = 4
    # n_embd: int = 64 - size of embedding (token) vectors
    # n_embd2: int = 64 - varies by model. For the MLP it's the hidden layer dim/width
    # n_head: int = 4

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd

        #input hidden layer
        self.weight1 = nn.Parameter(torch.randn((self.block_size * config.n_embd), config.n_embd2) * 0.1)
        self.bias1 = nn.Parameter(torch.zeros(config.n_embd2))

        #output layer
        self.weight2 = nn.Parameter(torch.randn(config.n_embd2, self.vocab_size) * 0.1)
        self.bias2 = nn.Parameter(torch.zeros(self.vocab_size))

        #so we have an input vector that has a (block_size * n_embd) size 
        #we want to take this vector and feed it into the first layer of our MLP. The first layer outputs n_embd2 values, and so our weights matrix needs to be (block_size * n_embd) x n_embd2
        #then this middle/final layer of our MLP, which has n_embd2 neurons, needs to map to 27 output values. And so this layer has 27 neurons in it. The final layer will have weights of size 
        #n_embd2 x 27. Don't forget the nonlinearity between the middle and final layers. 

        self.embeddings = nn.Parameter(torch.randn(self.vocab_size, config.n_embd))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # gather the word embeddings of the previous 3 words
        #remember idx is batched, so the input dims are B x T
        word_embeds = self.embeddings[idx] #should hopefully get all batches, all embeddings, so now we have B x T x n_embd
        
        temp = word_embeds
        for i in range(1, self.block_size):
            curr_roll = torch.roll(temp, i, dims= 1)
            word_embeds = torch.cat([word_embeds, curr_roll], dim=2) #we concat the context for each token onto that tokens n_embd dimension with torch.roll which shifts everything. Now we should have B x T x (block_size * n_embd)
        
        #we need to mask out tokens from the future in the word_embeds matrix, so the model can't cheat.
        #we want a T by block_size * n_embd mask matrix. We can broadcast this over all of our batches, and since we want to mask all n_embeds we change the shape
        rows = torch.arange(word_embeds.shape[1]).unsqueeze(1)
        cols = torch.arange(word_embeds.shape[2]).unsqueeze(0) // self.n_embd
        diff = rows - cols
        mask = (diff >= 0) #so a T x (block_size * n_embd) mask matrix. 
        word_embeds = word_embeds.masked_fill(~mask, 0) #same size as before just with tokens masked. Each token cannot attend to tokens more than block_size before it

        layer1_logit = torch.tanh(word_embeds @ self.weight1 + self.bias1) #so this should output a B x T x n_embd2 matrix

        layer2_logit = (layer1_logit @ self.weight2 + self.bias2) #so now we should have a B x T x vocab_size output

        if targets is not None:
            loss = F.cross_entropy(layer2_logit.flatten(0,1), targets.flatten(), ignore_index=-1) #targets is shape B T and logits is shape B T vocab_size, CE wants N x C where C is the num of classes and N is the number of examples. So we have BT examples, 
            #and for each example we have vocab_size classes. So, we want an input of BT V and a target of BT 
            return layer2_logit, loss
        return layer2_logit, None

# -----------------------------------------------------------------------------
# Bigram language model

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    # class ModelConfig:
    # block_size: int = None # length of the input sequences of integers
    # vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # # parameters below control the sizes of each model slightly differently
    # n_layer: int = 4
    # n_embd: int = 64
    # n_embd2: int = 64
    # n_head: int = 4

    #this is a trained bigram model, so we want a self.logits lookup table of size vocab_size x vocab_size
    #so for any item in the vocab, we can do logits[id] and it'll return a logit that we then sample from, presumably

    def __init__(self, config):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(config.vocab_size, config.vocab_size)) #this is our lookup table for logits
        
    def get_block_size(self):
        return 1

    def forward(self, idx, targets=None):
        # 'forward pass', lol
        logit = self.weights[idx] # shape B T V, targets shape B T 
        if targets is not None:
            loss = F.cross_entropy(logit.flatten(0, 1), targets.flatten(), ignore_index=-1)
            return logit, loss
        return logit, None

# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    #model is the model we're using. I assume model(x,y) gives us the logits and loss whereas model(x) just gives us a logit
    #and so idx is the batch of token seqs we're generating from
    #temperature determines the way we sample. How? 
    # What is do_sample? Is that just whether we sample deterministically? And what is top_k?
    model.eval()
    #indices are in idx and are b, t shape
    for _ in range(max_new_tokens):
        block_size = model.get_block_size()
        input = idx[:, -block_size:]

        logits, _ = model(input)
        logits = logits[:, -1, :] / temperature #we keep only the last logit, so keep all batches, get the last logit, keep all dims in the logit
        # B x v now

        if top_k is not None:
            values, _ = torch.topk(logits, top_k) #values will be B x top_k 
            mask = logits < values[:, [-1]]
            logits = logits.masked_fill(mask, -float('Inf')) #so logits is still B x v but with -inf values
        
        probs = F.softmax(logits, dim=-1) #this is per batch so b by v
        next_char = torch.multinomial(probs, 1) if do_sample else torch.argmax(probs, dim=-1, keepdim=True) #should give B x 1 now
        idx = torch.cat((idx, next_char), dim=1) #idx is B x t and next_char is B x 1, should work?
    
    return idx


def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        super().__init__()
        #words is a list of words to make the dataset with
        #chars is a list of all the possible characters
        #max_word_length is just the max len of a word in the dataset
        self.vocab = [0] + chars
        self.data = words
        self.max_word_length = max_word_length

        self.ctoi = {c:i for i, c in enumerate(self.vocab)}
        self.itoc = {i:c for i, c in enumerate(self.vocab)}

    def __len__(self):
        #must be len of the dataset
        return len(self.data)

    def contains(self, word):
        #whether our dataset has a word in it
        return word in self.data

    def get_vocab_size(self):
        #len of chars plus any special tokens. 
        return len(self.vocab)

    def get_output_length(self):
        #seems to be the max length of sequence seen in the training data + 1 for the EOS token
        #we don't do +2 because of a quirk in training
        return self.max_word_length + 1

    def encode(self, word):
        #encodes a word into an index for the current dataset, presumably in vocab
        return [self.ctoi[c] for c in word]

    def decode(self, ix):
        #returns a word from the vocab given an index
       return [self.itoc[i] for i in ix]

    def __getitem__(self, idx):
        #returns dataopoint x and label for it y at a specific index. pytorch function that enables the dataset[idx] operation. 
        #so presumably x is just the word. And since we're training character level models y is just the word but shifted over by one
        word = self.data[idx]
        word_encode = self.encode(word)

        X = [0] + word_encode + [0] * (self.max_word_length - len(word_encode))
        Y = word_encode + [0] + [-1] * (self.max_word_length - len(word_encode))

        return torch.tensor(X, dtype=torch.int64), torch.tensor(Y, dtype=torch.int64)
        

def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

