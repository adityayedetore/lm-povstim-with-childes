# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse


# Arguments to give to the arg parser
lm_parser = argparse.ArgumentParser(add_help=False)

lm_parser.add_argument('--data', type=str,
                       help='location of the data corpus')
lm_parser.add_argument('--load', type=str, default=None,
                       help='load model')

lm_parser.add_argument('--model', type=str, default='LSTM',
                       help='type of model (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
lm_parser.add_argument('--emsize', type=int, default=200,
                       help='size of word embeddings')
lm_parser.add_argument('--nhid', type=int, default=200,
                       help='number of hidden units per layer')
lm_parser.add_argument('--nlayers', type=int, default=2,
                       help='number of layers')
lm_parser.add_argument('--dropout', type=float, default=0.2,
                       help='dropout applied to layers (0 = no dropout)')
lm_parser.add_argument('--tied', action='store_true',
                       help='tie the word embedding and softmax weights')
lm_parser.add_argument('--nhead', type=int, default=2,
                       help='the number of heads in the encoder/decoder of the transformer model')

lm_parser.add_argument('--lr', type=float, default=20,
                       help='initial learning rate')
lm_parser.add_argument('--clip', type=float, default=0.25,
                       help='gradient clipping')
lm_parser.add_argument('--patience', type=int, default=2,
                       help='number of epochs without improvement to do before stopping')
lm_parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                       help='batch size')

lm_parser.add_argument('--bptt', type=int, default=35,
                       help='sequence length to backpropagate through')


lm_parser.add_argument('--seed', type=int, default=1111,
                       help='random seed')
lm_parser.add_argument('--cuda', action='store_true',
                       help='use CUDA')
lm_parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                       help='report interval')
lm_parser.add_argument('--save', type=str, default='model.pt',
                       help='path to save the final model')
lm_parser.add_argument('--log', type=str, default='log.txt',
                       help='path to logging file')
