#!/usr/bin/env python
# coding: utf-8

# # Childes Procsssing 

# In[1]:

def listify_data(raw_string):
    return [line.split() for line in raw_string.splitlines()]

def read_data(filename):
    with open(filename, "r") as f:
        raw_data = f.read()
    return raw_data


# In[2]:

def unlistify_data(data):
    zipped = [" ".join(line) for line in data]
    return "\n".join(zipped)

def write_data(data, filename):
    with open(filename, "w") as f:
        f.write(unlistify_data(data))


# ## Cleanup
# 
# Remove '\_' and sentences that only contain a single token (sentences that only have '.', or '?', etc.).

# In[3]:


def clean_and_listify(data):
    data_str = "\n".join([" ".join(s) for s in data])
    new_data = data_str.replace("_"," ")
    new_data = listify_data(new_data)
    removed_empty = []
    for sent in new_data:
        if len(sent) > 1:
            removed_empty.append(sent)
    return removed_empty


# ## Split possesives and contractions
# 
# Seperate possesives: "camel's" -> "camel 's"
# 
# Also seperate contractions. The subwords _n't_ , _'re_ , _'ll_ , _'m_ , _'ve_ , and _'d_ should be prepened with spaces. 

# In[4]:


# this takes a minute and a half to run on my machine

def split_possesives_and_contractions(word):
    if word.endswith("'s"):
        return word[:-2] + " 's"
    if word == "can't":
        return "can n't"
    if word.endswith("n't"):
        return word[:-3] + " n't"
    if word.endswith("'re"):
        return word[:-3] + " 're"
    if word.endswith("'m"):
        return word[:-2] + " 'm"
    if word.endswith("'d"):
        return word[:-2] + " 'd"
    if word.endswith("'ll"):
        return word[:-3] + " 'll"
    if word.endswith("'ve"):
        return word[:-3] + " 've"
    if word.endswith("s'"):
        return word[:-1] + " '"
    if word.endswith("'r"):
        return word[:-2] + " are"
    if word.endswith("'has"):
        return word[:-4] + " has"
    if word.endswith("'is"):
        return word[:-3] + " is"
    if word.endswith("'did"):
        return word[:-4] + " did"
    if word == "wanna":
        return "want to"
    if word == "hafta":
        return "have to"
    if word == "gonna":
        return "going to"
    if word == "okay":
        return "ok"
    if word == "y'all":
        return "you all"
    if word == "c'mere":
        return "come here"
    if word == "I'ma":
        return "I am going to"
    if word == "what'cha":
        return "what are you"
    if word == "don'tcha":
        return "do you not"
    
    
    # List of startswith exceptions: ["t'", "o'", "O'", "d'"]
    # List of == exceptions: ["Ma'am", "ma'am", "An'", "b'ring", "Hawai'i","don'ting", "rock'n'roll" "don'ting", "That'scop","that'ss","go'ed", "s'pose", "'hey", "me'", "shh'ell", "th'do", "Ross'a", "him'sed"] 
    # List of in exceptions: ["_", "-"]
    # List of endswith exceptions (note that this one is a catch all condition): ["'"]

    return word

def split_line(line):
    s = [split_possesives_and_contractions(word) for word in line]
    return " ".join(s).split()

def split_data(data):
    return [split_line(line) for line in data]


# ## Unking
# 
# Replace infrequent words with `<unk>` tokens. 
# 
# Note that the unked tokens are based on the training set, even for the validation and test sets.

# In[5]:


def count_frequencies(data):
    frequencies = {}
    for line in data:
        for word in line:
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1
    return frequencies

# words with frequency > cutoff
def make_vocab(data, cutoff):
    frequencies = count_frequencies(data)
    high_frequency_tokens = set()
    for token in frequencies:
        if frequencies[token] > cutoff:
            high_frequency_tokens.add(token)
    return high_frequency_tokens

def unk(data, vocab):
    unked_data = []
    for line in data:
        unked_line = []
        for word in line:
            if word in vocab:
                unked_line.append(word)
            else:
                unked_line.append("<unk>")
        unked_data.append(unked_line) 
    return unked_data


# In[ ]:


def clean_and_unk(train, valid, test, excluded):
    # Clean all the datasets
    train_cleaned = clean_and_listify(train)
    valid_cleaned = clean_and_listify(valid)
    test_cleaned = clean_and_listify(test)
    excluded_cleaned = clean_and_listify(excluded)

    # Split possessives and contractions in all the datasets
    train_split = split_data(train_cleaned)
    valid_split = split_data(valid_cleaned)
    test_split = split_data(test_cleaned)
    excluded_split = split_data(excluded_cleaned)

    # Create the vocab: All words that occurs more than 2 times
    # in the training set
    train_vocab = make_vocab(train_split, cutoff=2)

    # unk the train, valid, and test data
    train_unked = unk(train_split, train_vocab)
    valid_unked = unk(valid_split, train_vocab)
    test_unked  = unk(test_split,  train_vocab)
    vocab = list(train_vocab) + ["<unk>"]

    return train_unked, valid_unked, test_unked, excluded_split, vocab

