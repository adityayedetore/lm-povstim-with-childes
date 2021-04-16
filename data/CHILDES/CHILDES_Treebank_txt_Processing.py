#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def edit_tokenization(data_string):
    corpus_string_edited = data_string.replace(" 't ", " n't ")
    for letter in "abcdefghijklmnopqrstuvwxyz":
        corpus_string_edited = corpus_string_edited.replace(" " + letter + "l ", " " + letter + " ")
        corpus_string_edited = corpus_string_edited.replace("\n" + letter + "l ", "\n" + letter + " ")
    corpus_string_edited = corpus_string_edited.replace(" cha ", " you ")
    corpus_string_edited = corpus_string_edited.replace("\ncha ", "\nyou ")
    corpus_string_edited = corpus_string_edited.replace("okay ", "ok ")
    corpus_string_edited = corpus_string_edited.replace("hmm ", "hm ")
    corpus_string_edited = corpus_string_edited.replace("will n't ", "wo n't ")
    corpus_string_edited = corpus_string_edited.replace("ING", "ing")
    return corpus_string_edited 

def to_alnum(string):
    return ''.join(e.lower() for e in string if e.isalnum() or e == "\n")

def split_treebank(excluded, decl, quest):
    excluded_edited = edit_tokenization(excluded)

    # Give us a list of all sentences to be excluded, formatted in a way that
    # removes punctuation etc. so we don't miss matches due to tokenization details
    excluded_alnum = set(to_alnum(excluded_edited).replace('xxx','').splitlines())

    decl = [edit_tokenization(d) for d in decl]
    quest = [edit_tokenization(q) for q in quest]

    test = []
    not_test = []

    train_size = int(0.8 * len(decl))
    test_size = int(0.1 * len(decl))
    len_test = 0

    for i,q in enumerate(quest):
        # If the question is in the excluded set, then it can't be in
        # training or validation
        if to_alnum(q) in excluded_alnum and len_test < test_size:
            test.append(decl[i] + " " +   quest[i] + "\n")
            len_test += 1
        else:
            not_test.append(decl[i] + " " +   quest[i] + "\n")

    train, valid = not_test[:train_size], not_test[train_size:]

    return train, valid, test

