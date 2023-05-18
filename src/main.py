# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import time

import torch
import torch.nn as nn

from dictionary_corpus import Corpus
import model
from lm_argparser import lm_parser
from utils import repackage_hidden, get_batch, batchify, batchify_finetuning

parser = argparse.ArgumentParser(parents=[lm_parser],
                                 description="Basic training and evaluation for RNN LM")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

logging.info("Loading data")
start = time.time()
corpus = Corpus(args.data)
logging.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
logging.info("Vocab size %d", ntokens)
eval_batch_size = 10

logging.info("Batchifying..")
if args.finetune:
    train_data = batchify_finetuning(corpus.train, args.batch_size, corpus.dictionary.word2idx['?'], args.cuda, padding_id =0)
    val_data = batchify_finetuning(corpus.valid, eval_batch_size, corpus.dictionary.word2idx['?'], args.cuda, padding_id =0)
    test_data = batchify_finetuning(corpus.test, eval_batch_size, corpus.dictionary.word2idx['?'], args.cuda, padding_id =0)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
else:
    train_data = batchify(corpus.train, args.batch_size, args.cuda)
    val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
    test_data = batchify(corpus.test, eval_batch_size, args.cuda)
    criterion = nn.CrossEntropyLoss() 



###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")

if args.load == None:
    if args.model == "Transformer":
        model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    else:
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
else:
    with open(args.load, 'rb') as f:
        if args.cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location = lambda storage, loc: storage)

if args.cuda:
    model.cuda()



###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if args.model != "Transformer":
        hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                output = output.view(-1, ntokens)
                hidden = repackage_hidden(hidden)

            total_loss += len(data) * nn.CrossEntropyLoss()(output, targets).item()

    return total_loss / (len(data_source) - 1)


def evaluate_finetuned(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch, b in enumerate(data_source):
            data, targets = b[:,:-1].T, b[:,1:].T.flatten()

            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = model.init_hidden(len(b))
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                output = output.view(-1, ntokens)

            total_loss +=  criterion(output, targets).item()

    return total_loss / (len(data_source) - 1)

def finetune():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    # Loop over training set in chunks of decl quest pairs
    for batch, b in enumerate(train_data):
        data, targets = b[:,:-1].T, b[:,1:].T.flatten()
       
        model.zero_grad()
        
        if args.model == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            # truncated BPP
            hidden = model.init_hidden(len(b))
            hidden = repackage_hidden(hidden) # is this necessary? 
            output, hidden = model(data, hidden)
            output = output.view(-1, ntokens)

        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    if args.model != "Transformer":
        hidden = model.init_hidden(args.batch_size)

    # Loop over training set in chunks of size args.bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
       
        model.zero_grad()
        
        if args.model == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            # truncated BPP
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    patience_exhausted = False
    epochs_since_improved = 0
    epoch = 0

    while not patience_exhausted:
        epoch_start_time = time.time()

        if args.finetune:
            finetune()
            val_loss = evaluate_finetuned(val_data)
        else:
            train()
            val_loss = evaluate(val_data)

        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            epochs_since_improved = 0
        else:
            epochs_since_improved += 1
            if epochs_since_improved >= args.patience:
                patience_exhausted = True

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            logging.info('| epochs since loss improved: ' + str(epochs_since_improved))
            logging.info('| reducing learning rate to ' + str(lr))

            # Return to the best saved model checkpoint
            with open(args.save, 'rb') as f:
                model = torch.load(f)
        
        logging.info('-' * 89)
        epoch += 1

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
if args.finetune:
    test_loss = evaluate_finetuned(test_data)
else:
    test_loss = evaluate(test_data)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging.info('=' * 89)
