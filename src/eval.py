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

from dictionary_corpus import Corpus, tokenize
import model
from utils import repackage_hidden, batchify_finetuning, batchify, get_batch

parser = argparse.ArgumentParser(description='Evaludate finetuned model')
parser.add_argument('--data', type=str, required=True,
        help='path to data to evaluate on')
parser.add_argument('--model', type=str, default="model.pt",
        help="path to model to evaluate")
parser.add_argument('--results', type=str, default="results.txt",
        help="path to save results")
parser.add_argument('--log', type=str, default="log.txt",
        help="path to logging file")
parser.add_argument('--finetuning_data', type=str, required=True,
        help='path to finetuning data')
parser.add_argument('--rnn', action='store_true',
        help='evaluating a RNN')
parser.add_argument('--seed', type=int, default=999,
        help='random seed')
parser.add_argument('--batch_size', type=int, default=500,
        help='batch size')
parser.add_argument('--log_interval', type=int, default=10,
        help='log interval')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--print', action='store_true',
        help='print sentences and results')
parser.add_argument('--ppl', action='store_true',
        help='get sentence ppls')
parser.add_argument('--recall', type=int, default=0,
        help='get recall@k')
parser.add_argument('--babyberta', action='store_true',
        help='evaluate babyberta')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.basicConfig(format='%(message)s')
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

#logging.info("Loading data")
start = time.time()
corpus = Corpus(args.finetuning_data)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word
#logging.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
#logging.info("Vocab size %d", ntokens)

if not args.ppl and not args.recall:
    if args.data.endswith(".data"):
        #logging.info("THIS DATA FILE SHOULD HAVE THE FIRST AUX AND THE MAIN AUX OF THE SENTENCES LISTED AFTER THE QUESTION and before the ?. ELSE THIS CODE WON'T WORK.")
        pass
    else:
        logging.info("YOU SHOULD BE EVALUATING WITH THE .DATA FILE!")
else: 
    if args.data.endswith(".data"):
        pass
        #logging.info("You probably don't want to evaluate the perplexity on the .data file")


eval_corpus = tokenize(corpus.dictionary, args.data)

#logging.info("Batchifying..")

with open(args.data) as f:
    if args.babyberta:
        split_key = 'Maria'
        split_id = corpus.dictionary.word2idx[split_key]
        eval_data = batchify_finetuning(eval_corpus, args.batch_size, split_id=split_id, cuda=args.cuda, padding_id=0)
    elif args.recall:
        eval_data = batchify(eval_corpus, args.batch_size)
    else:
        eval_data = batchify_finetuning(eval_corpus, args.batch_size, corpus.dictionary.word2idx['?'], args.cuda, padding_id =0)

criterion = nn.CrossEntropyLoss(ignore_index=0) 

###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")

with open(args.model, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        model = torch.load(f, map_location = lambda storage, loc: storage)

if args.cuda:
    model.cuda()



###############################################################################
# Eval code
###############################################################################
def evaluate_ppl(data_source):
    model.eval()

    ppls = []
    with torch.no_grad():
        for batch, b in enumerate(data_source):
            data, targets = b.T[:-1], b.T[1:]
            if not args.rnn:
                output = model(data)
            else:
                hidden = model.init_hidden(len(b))
                output, hidden = model(data, hidden)

            ppls += [torch.exp(criterion(o, t)).item() for o,t in zip(output.permute(1,0,2),targets.T)]
            if batch % args.log_interval == 0 and batch > 0:
                pass
                #logging.info(str(batch * args.batch_size) + "/" + str(sum([d.shape[0] for d in data_source])) + " sentences done")
    return ppls

def evaluate(data_source):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch, b in enumerate(data_source):
            data = b.T
            if not args.rnn:
                output = model(data)
            else:
                hidden = model.init_hidden(len(b))
                output, hidden = model(data, hidden)

            preds += torch.argmax(output, dim=2).T.tolist()
            if batch % args.log_interval == 0 and batch > 0:
                pass
                #logging.info(str(batch * args.batch_size) + "/" + str(sum([d.shape[0] for d in data_source])) + " sentences done")
    return preds

def evaluate_recall(data_source):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch, i in enumerate(range(0, data_source.size(0) - 1, args.batch_size)):
            data, targets = get_batch(data_source, i, args.batch_size)
            if not args.rnn:
                output = model(data)
            else:
                hidden = model.init_hidden(len(data))
                output, hidden = model(data, hidden)

            preds += torch.topk(output, k=args.recall, dim=2)[1].permute(1,0,2).reshape(-1, args.recall).tolist()
    return preds

def accuracy(preds, targets):
    with torch.no_grad():
        move_first = []
        move_main = []
        pred_auxs = []
        for (pred, target) in zip(preds, targets):
            pred_aux = idx2word[pred[target.index(corpus.dictionary.word2idx['.'])]] #note that the predictions are shifted by one compared to the targets (since there is no prediction for the first word), which is why we don't add one to the index here
            actual_first_aux = idx2word[int(target[-3])]
            actual_main_aux = idx2word[int(target[-2])]
            if args.print:
                print("predicted: " + ' '.join([idx2word[int(x)] for x in pred]))
                print("target: " + ' '.join([idx2word[int(x)] for x in target]))
                print("predicted aux: " + pred_aux)
                print("actual aux: " + actual_main_aux)
            move_first.append(int(pred_aux == actual_first_aux))
            move_main.append(int(pred_aux == actual_main_aux))
            pred_auxs.append(pred_aux)
        return move_first, move_main, pred_auxs

if not args.recall:
    eval_data_flat = [x[:x.index(0)] if 0 in x else x for y in eval_data for x in y.tolist()]
if args.babyberta: 
    results = evaluate_ppl(eval_data)
    with open(args.results, 'w') as fo:
        fo.write('\n'.join(str(ppl) for ppl in results))
    comp_size = 2
    d = {i:0 for i in range(comp_size)}
    for i in range(0, len(results), comp_size):
        group = results[i:i+comp_size]
        max_pos = group.index(min(group))
        d[max_pos] += 1
    logging.info('\t'.join([str(x).ljust(10) for x in d.values()]))
elif args.ppl and not args.slor:
    results = evaluate_ppl(eval_data)
    with open(args.results, 'w') as fo:
        fo.write('\n'.join(str(ppl) for ppl in results))
    #logging.info('=' * 89)
    #logging.info('Maria End of Evaluation |')
    #logging.info('=' * 89)
    comp_size = 6
    d = {i:0 for i in range(comp_size)}
    for i in range(0, len(results), comp_size):
        group = results[i:i+comp_size]
        max_pos = group.index(min(group))
        d[max_pos] += 1
    logging.info('\t'.join([str(x).ljust(10) for x in d.values()]))
elif args.ppl and args.slor:
    results = evaluate_ppl(eval_data)
    with open(args.results, 'w') as fo:
        fo.write('\n'.join(str(ppl) for ppl in results))
    with open (args.slor) as f:
        slors = [float(i) for i in f.readlines()]
    d = {0:0,1:0,2:0,3:0,4:0,5:0}
    for i in range(0, len(results), 6):
    #for i in range(0, len(results), 2):
        group = [r - slors[i+j] for j,r in enumerate(results[i:i+6])]
        #group = [r for j,r in enumerate(results[i:i+6])]
        #group = [r for j,r in enumerate(results[i:i+2])]
        max_pos = group.index(min(group))
        d[max_pos] += 1
    logging.info('\t'.join([str(x) for x in d.values()]))
elif args.recall:
    all_results = evaluate_recall(eval_data)
    count = 0
    total = 0
    print([[idx2word[j] for j in i] for i in all_results])
    print([idx2word[i] for i in eval_data.T.reshape(-1)])
    for actual, recall in zip(eval_data.T.reshape(-1), all_results):
        count += int(actual.item() in recall)
        print(idx2word[actual.item()], '|'.join([idx2word[i] for i in recall]))
        total += 1
    logging.info('the recall@k is: ' + str(count/total))
else:
    results = evaluate(eval_data)
    move_first,move_main,pred_auxs = accuracy(results, eval_data_flat)

    with open(args.results, 'w') as fo:
        #fo.write("0 and 1 mean incorrect and correct move first, 2 and 3 mean incorrect and correct move main\n")
        for r,d,f,m,p in zip(results, eval_data_flat, move_first, move_main,pred_auxs):
            fo.write("target:\t" + str(f) + "\t" + str(m + 2) + "\t" +  ' '.join([corpus.dictionary.idx2word[word] for word in d]) + "\n")
            fo.write("predicted:\t" + str(f) + "\t" + str(m + 2) + "\t" +  ' '.join([corpus.dictionary.idx2word[word] for word in r]) + " " + p + " " + "\n\n")

    #logging.info('=' * 89)
    #logging.info('| End of Evaluation | Move First proportion ' + str(sum(move_first)/len(move_first)) + '| Move Main proportion ' + str(sum(move_main)/len(move_main)))
    #logging.info('=' * 89)

    num_full_correct = 0
    num_full_correct_move_first = 0
    for r,d,f,m,p in zip(results, eval_data_flat, move_first, move_main,pred_auxs):
        i = d.index(word2idx['.']) + 1
        j = len(d) - 3
        num_full_correct += int(d[i:j] == r[i-1:j-1])
        i = d.index(word2idx['.']) 
        move_first_question = [d[j]] + d[:d.index(d[j])] + d[d.index(d[j])+1:i]
        num_full_correct_move_first += int(move_first_question == r[i:j-1])
        #if move_first_question == r[i-1:j-1]: 
            #print(' '.join([idx2word[x] for x in r[i-1:j-1]]))

    #logging.info('=' * 89)
    #logging.info('| End of Evaluation | Full sent correct with move main proportion ' + str(num_full_correct/len(results)))
    #logging.info('| End of Evaluation | Full sent correct with move firt proportion ' + str(num_full_correct_move_first/len(results)))
    #logging.info('=' * 89)
    logging.info('\nProportion that the first word follows MOVE-FIRST: ' + (str(sum(move_first)/len(move_first)) + '0'*6)[:6] + '\nProportion that the first word follows MOVE-MAIN: ' + (str(sum(move_main)/len(move_main)) + '0'*6)[:6]) #+ '\nPropostion that the full sentence follows MOVE-MAIN: ' + (str(sum(move_main)/len(move_main)) + '0'*6)[:6] + '\nProportion that the full sentence follows MOVE-FIRST: ' + (str(num_full_correct/len(results)) + '0'*6)[:6] + '\t' + (str(num_full_correct_move_first/len(results)) + '0'*6)[:6])

