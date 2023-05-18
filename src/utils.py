# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence

# Detaches hidden states from their history,
# to avoid backpropagating when you don't want to
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

def batchify_finetuning(data, batch_size, split_id, cuda, padding_id=0):
    def is_split_id(x):
        return x == split_id
    data = [torch.tensor(list(group) + [split_id]) for k, group in groupby(data.tolist(), is_split_id) if not k]
    chunks = [data[x:x+batch_size] for x in range(0, len(data), batch_size)]
    data = [pad_sequence(c, batch_first=True, padding_value=padding_id) for c in chunks]
    if cuda:
        data = [d.cuda() for d in data]
    return data

