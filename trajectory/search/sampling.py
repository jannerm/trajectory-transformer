import numpy as np
import torch
import pdb

#-------------------------------- helper functions --------------------------------#

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def filter_cdf(logits, threshold):
    batch_inds = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    bins_inds = torch.arange(logits.shape[-1], device=logits.device)
    probs = logits.softmax(dim=-1)
    probs_sorted, _ = torch.sort(probs, dim=-1)
    probs_cum = torch.cumsum(probs_sorted, dim=-1)
    ## get minimum probability p such that the cdf up to p is at least `threshold`
    mask = probs_cum < threshold
    masked_inds = torch.argmax(mask * bins_inds, dim=-1)
    probs_threshold = probs_sorted[batch_inds, masked_inds]
    ## filter
    out = logits.clone()
    logits_mask = probs <= probs_threshold.unsqueeze(dim=-1)
    out[logits_mask] = -1000
    return out

def round_to_multiple(x, N):
    '''
        Rounds `x` up to nearest multiple of `N`.

        x : int
        N : int
    '''
    pad = (N - x % N) % N
    return x + pad

def sort_2d(x):
    '''
        x : [ M x N ]
    '''
    M, N = x.shape
    x = x.view(-1)
    x_sort, inds = torch.sort(x, descending=True)

    rows = inds // N
    cols = inds % N

    return x_sort, rows, cols

#-------------------------------- forward pass --------------------------------#

def forward(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    '''
        A wrapper around a single forward pass of the transformer.
        Crops the input if the sequence is too long.

        x : tensor[ batch_size x sequence_length ]
    '''
    model.eval()

    block_size = min(model.get_block_size(), max_block or np.inf)

    if x.shape[1] > block_size:
        assert allow_crop, (
            f'[ search/sampling ] input size is {x.shape} and block size is {block_size}, '
            'but cropping not allowed')

        ## crop out entire transition at a time so that the first token is always s_t^0
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]

    logits, _ = model(x, **kwargs)

    return logits

def get_logp(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## apply softmax to convert to probabilities
    logp = logits.log_softmax(dim=-1)

    return logp

#-------------------------------- sampling --------------------------------#

def sample(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        Samples from the distribution parameterized by `model(x)`.

        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## keep track of probabilities before modifying logits
    raw_probs = logits.softmax(dim=-1)

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## apply softmax to convert to probabilities
    probs = logits.softmax(dim=-1)

    ## sample from the distribution
    ## [ batch_size x 1 ]
    indices = torch.multinomial(probs, num_samples=1)

    return indices, raw_probs

@torch.no_grad()
def sample_n(model, x, N, **sample_kwargs):
    batch_size = len(x)

    ## keep track of probabilities from each step;
    ## `vocab_size + 1` accounts for termination token
    probs = torch.zeros(batch_size, N, model.vocab_size + 1, device=x.device)

    for n in range(N):
        indices, p = sample(model, x, **sample_kwargs)

        ## append to the sequence and continue
        ## [ batch_size x (sequence_length + n) ]
        x = torch.cat((x, indices), dim=1)

        probs[:, n] = p

    return x, probs
