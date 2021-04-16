#!/usr/bin/env python
# coding: utf-8

# Instructions to read the XML version of the CHILDES corpus adapted from on the [nltk website](http://www.nltk.org/howto/childes.html). 
# 
# XML corpora can be downloaded from the [childes website](https://childes.talkbank.org/data-xml/Eng-NA/)

# In[1]:


import nltk
from childes import CHILDESCorpusReader # Edited version of nltk.corpus.reader
from collections import defaultdict
import random
from random import sample
import copy


# In[2]:


def read_corpora(path_to_corpora, corpora_file_name):
    return CHILDESCorpusReader(path_to_corpora, corpora_file_name + "/.*.xml")


# In[3]:


def map_files_to_non_target_child_utterances(corpora):
    filtered_corpora = {}
    for fileid in corpora.fileids():
        participants = get_non_target_child_participants(corpora, fileid)
        utterances = get_utterances_filtered_by_participants(corpora, fileid, participants)
        if utterances != []:
            filtered_corpora[fileid] = utterances
    return filtered_corpora

def get_non_target_child_participants(corpora, fileid):
    non_target_child_participants = []
    corpora_participants = corpora.participants(fileid)
    for participants in corpora_participants:
        for key in participants.keys():
            dct = participants[key]
            if dct['role'] != "Target_Child":
                non_target_child_participants.append(dct['id'])
    return non_target_child_participants

def get_utterances_filtered_by_participants(corpus, fileid, participants):
    utterances = corpus.sents(fileid, speaker=participants, replace=True) # replace=True
    cleaned_utts = [utt for utt in utterances if utt != []]
    return cleaned_utts


# In[4]:


def is_treebank_file(fileid):
    for treebank_corpus_name in ['Brown','Soderstrom','Valian','Suppes']:
        if treebank_corpus_name in fileid:
            return True
    return False

def split_treebank(files_to_utterances):
    treebank = {file : files_to_utterances[file] for file in files_to_utterances if is_treebank_file(file)}
    not_treebank = {file : files_to_utterances[file] for file in files_to_utterances if not is_treebank_file(file)}
    return treebank, not_treebank

def count_questions(sents):
    return len([sent for sent in sents if sent[-1] == '?'])

def sort_dict_by_number_of_questions(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: count_questions(item[1]))}

def hold_out(files_to_utterances):
    files_to_utterances_sorted = sort_dict_by_number_of_questions(files_to_utterances)
    included = copy.deepcopy(files_to_utterances_sorted)
    excluded = {}
    for i,file in enumerate(files_to_utterances_sorted):
        if i % 10 == 0:
            excluded[file] = included.pop(file)
    return included, excluded


# In[5]:


def train_valid_test_split(files_to_utterances):
    files_to_utterances_sorted = sort_dict_by_value_length(files_to_utterances)
    utterances = [utts for utts in files_to_utterances_sorted.values()]
    train, valid, test = [],[],[]
    count = 0
    while count < len(utterances) - 100:
        sample_indices = sample(range(count, count + 100), 100)
        for i in sample_indices[0:90]:
            train += utterances[i]
        for i in sample_indices[90:95]:
            valid += utterances[i]
        for i in sample_indices[95:100]:
            test += utterances[i]
        count += 100
    return train, valid, test

def sort_dict_by_value_length(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: len(item[1]))}


# In[6]:


def remix_held_out(valid, test, excluded):
    excluded_utterances = [utt for utts in excluded.values() for utt in utts]
    excluded_size = len(excluded_utterances)
    reshuffle_size = int(excluded_size/2)
    return valid + test[:reshuffle_size], test[reshuffle_size:] + excluded_utterances



# In[8]:


def process_childes_xml(path_to_childes="./", childes_file_name="childes-xml"):
    corpora = read_corpora(path_to_corpora=path_to_childes, corpora_file_name=childes_file_name)
    files_to_utterances = map_files_to_non_target_child_utterances(corpora)
    treebank, not_treebank = split_treebank(files_to_utterances)
    included_treebank, excluded = hold_out(treebank)
    included = {**not_treebank, **included_treebank} # Python 3.5 or greater
    train, valid, test = train_valid_test_split(included)
    valid_remixed, test_remixed = remix_held_out(valid, test, excluded)
    excluded = [utt for utts in excluded.values() for utt in utts]
    return train, valid_remixed, test_remixed, excluded

