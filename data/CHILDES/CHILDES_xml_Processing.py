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


# Create a CHILDESCorpusReader object
def read_corpora(path_to_corpora, corpora_file_name):
    return CHILDESCorpusReader(path_to_corpora, corpora_file_name + "/.*.xml")


# Input: CHILDESCorpusReader object
# Output: A dict whose keys are the fileids from the
#         CHILDESCorpusReader's corpus, and whose values
#         are a list of utterances in that file made by
#         any participant except the target child
def map_files_to_non_target_child_utterances(corpora):
    filtered_corpora = {}
    for fileid in corpora.fileids():
        participants = get_non_target_child_participants(corpora, fileid)
        utterances = get_utterances_filtered_by_participants(corpora, fileid, participants)
        if utterances != []:
            filtered_corpora[fileid] = utterances
    return filtered_corpora

# Returns list of participant IDs for participants who
# are not the target child
def get_non_target_child_participants(corpora, fileid):
    non_target_child_participants = []
    corpora_participants = corpora.participants(fileid)
    for participants in corpora_participants:
        for key in participants.keys():
            dct = participants[key]
            if dct['role'] != "Target_Child":
                non_target_child_participants.append(dct['id'])
    return non_target_child_participants

# Returns utterances from `fileid` in `corpus` spoken by any
# participant in `participants`
def get_utterances_filtered_by_participants(corpus, fileid, participants):
    utterances = corpus.sents(fileid, speaker=participants, replace=True) # replace=True
    cleaned_utts = [utt for utt in utterances if utt != []]
    return cleaned_utts


# Checks if fileid is in the treebank
def is_treebank_file(fileid):
    for treebank_corpus_name in ['Brown','Soderstrom','Valian','Suppes']:
        if treebank_corpus_name in fileid:
            return True
    return False

# Split all the utterances into those that are
# in the treebank and not in the treebank
def split_treebank(files_to_utterances):
    treebank = {file : files_to_utterances[file] for file in files_to_utterances if is_treebank_file(file)}
    not_treebank = {file : files_to_utterances[file] for file in files_to_utterances if not is_treebank_file(file)}
    return treebank, not_treebank

# Input: List of sentences
# Output: Number of questions in the list 
def count_questions(sents):
    return len([sent for sent in sents if sent[-1] == '?'])

# Input: Dict whose keys are fileids and values are lists of utterances
#        in the file
# Output: Same dict but sorted by the number of questions in the file
def sort_dict_by_number_of_questions(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: count_questions(item[1]))}

# Given a files_to_utterances dictionary, split into 2 splits:
# an `included` split containing 90% of the data, and `excluded` 
# containing 10% of the data.
# Works by sorting the files by number of questions and then excluding 
# every 10th file so that the excluded contains only entire files and 
# contains approximately 10% of the questions
def hold_out(files_to_utterances):
    files_to_utterances_sorted = sort_dict_by_number_of_questions(files_to_utterances)
    included = copy.deepcopy(files_to_utterances_sorted)
    excluded = {}
    for i,file in enumerate(files_to_utterances_sorted):
        if i % 10 == 0:
            excluded[file] = included.pop(file)
    return included, excluded


# Split a files_to_utterances dict into training, validation,
# and test splits.
# Training: 90%, valid: 5%, test: 5%
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

# Sort a dictionary by the lengths of its values
def sort_dict_by_value_length(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: len(item[1]))}


# Reshuffle the validation and test data to add the treebank test data
# (aka `excluded`) to the test set, and then move some elements from `test`
# into `valid` so that valid and test are still the same size
def remix_held_out(valid, test, excluded):
    excluded_utterances = [utt for utts in excluded.values() for utt in utts]
    excluded_size = len(excluded_utterances)
    reshuffle_size = int(excluded_size/2)
    return valid + test[:reshuffle_size], test[reshuffle_size:] + excluded_utterances



def process_childes_xml(path_to_childes="./", childes_file_name="childes-xml"):
    # Create corpus reader
    corpora = read_corpora(path_to_corpora=path_to_childes, corpora_file_name=childes_file_name)

    # Get utterances from all participants other than target child
    files_to_utterances = map_files_to_non_target_child_utterances(corpora)

    # Split the utterances into those from the treebank and
    # not from the treebank
    treebank, not_treebank = split_treebank(files_to_utterances)

    # Split the treebank into included and excluded splits
    included_treebank, excluded = hold_out(treebank)

    # The full set of items included in pretraining is those
    # that are not in the treebank and those that are in the
    # included_treebank split
    included = {**not_treebank, **included_treebank} # Python 3.5 or greater

    # Split the pretraining data into train, valid, and test
    train, valid, test = train_valid_test_split(included)

    # Add the excluded treebank data to the second half of `test` to make the
    # final test set, and then combind `valid` with the first half of `test` to
    # make the final validation set
    valid_remixed, test_remixed = remix_held_out(valid, test, excluded)

    # The full list of excluded treebank uterances
    excluded = [utt for utts in excluded.values() for utt in utts]

    return train, valid_remixed, test_remixed, excluded

