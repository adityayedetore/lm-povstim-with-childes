#!/usr/bin/env python
# coding: utf-8

# NLTK tools for processing trees. You will need to have NLTK installed 
# to import this.
from nltk.tree import *
# Example of the NLTK tools
Tree.fromstring("(ROOT (NP (N John)) (VP (V slept)))")

def process_childes_treebank(path_to_treebank="childes-treebank"):
    # All the treebank files that are not
    # just about wh-questions

    folder_name = path_to_treebank
    if not folder_name.endswith("/"):
        folder_name = folder_name + "/"

    filenames = ["brown-adam.parsed",
                 "brown-eve+animacy+theta.parsed",
                 "brown-sarah.parsed",
                 "soderstrom.parsed",
                 "suppes.parsed",
                 "valian+animacy+theta.parsed",
                 "hslld-hv1-er.parsed",
                 "hslld-hv1-mt.parsed"]

    fis = [open(folder_name + filename, "r") for filename in filenames]

    trees = []
    tree = ""

    # Iterate over all the files
    for fi in fis:
        # Iterate over the lines in this file to add trees to the list of trees
        for line in fi:
            # If you have completed a tree, add it to the list of trees
            if line.strip() == "":
                if tree.strip() != "" and tree.count(")") == tree.count("(") and tree.count("ROOT") == 1:
                    trees.append(tree)
                tree = ""
            else:
                tree += line.strip()


    # In[3]:


    treesPrime = []

    # Loop over the trees
    for tree in trees:
        
        # Filter out incomplete trees that don't start and end with parentheses
        if tree[-1] == ")" and tree[0] == "(":
            # Get the list of words in the tree, i.e. its leaves
            words = (Tree.fromstring(tree)).leaves()
            
            # Now we filter the words, because the treebank contains some
            # null elements we want to ignore
            wordsFiltered = []
            for word in words:
                if "*" not in word:
                    wordsFiltered.append(word)
            
            # A couple trees have an extraneous slash so ignore those
            if "\\" not in tree:
                treesPrime.append(tree)


    # De-question-ify a question; that is, turn a 
    # question into a declarative. 
    def deQuest(tree):
        
        result = []
        
        # Check for a couple of cases that don't work
        good = 1
        if len(tree[0]) != 2:
            good = 0
        elif tree[0][0].label() != "VP" and tree[0][1].label() != ".":
            good = 0
            
        # This for loop is where the heavy lifting happens.
        # It is based on the NLTK tree methods, building on some
        # analysis of how the relevant trees are structured in the treebank.
        # But to understand what is going on you would probably need to work through 
        # some examples. The key thing is this:
        # - in the input tree, the auxiliary is at the top of the tree
        # - in the output, we want the auxiliary to be returned to 
        #   its rightful place within the sentence
        # One case to watch out for: Contractions are represented as 2 words,
        # and you have to move both words. e.g., "do n't you think so ?" -> "you do n't think so ."
        for child in tree[0]:
            if child.label() == "VP":
                toAdd = []
                added = 0
                if "NP" in child[0].label():
                    good = 0
                for kid in child:
                    if "NP" in kid.label() and not added:
                        result += kid.leaves()
                        result += toAdd
                        added = 1
                    elif not added:
                        toAdd += kid.leaves()
                    else:
                        result += kid.leaves()
                    
            else:
                result += child.leaves()
                
        if good: 
            return result[:-1] + ["."]
        else:
            return "INVALID QUESTION"
        


    # In[11]:


    # In this block, we generate input-output pairs from all
    # of our sentences. If the sentence is a yes-no question,
    # the input is the de-questionified version of the question,
    # while the output is the question. Otherwise, the input and
    # the output are both the sentence. 
    counter = 0
    countQ = 0
    total = 0
    countIntj = 0
    countFrag = 0
    decl = []
    quest = []


    for tree in treesPrime:
        okay = True

        # Ignore trivial trees
        if tree.count("(") > 2:
            total += 1
            newTree = Tree.fromstring(tree) 
            
            # If it's a valid yes-no question, the child(ren) of the
            # root will be phrases not strings
            for child in newTree[0]:
                if isinstance(child, str):
                    okay = False
            if okay:
                
                labels = [child.label() for child in newTree[0]]
                
                # Checking if it's a yes-no question: If there is just one child
                # of the root with a label SQ
                if len(newTree) == 1 and newTree[0].label() == "SQ":
                    countQ += 1
                    treeEdit = []
                    
                    # This is to get rid of the unpronounced words that are in some
                    # of the trees
                    for node in newTree.leaves():
                        if "*" not in node:
                            treeEdit.append(node)
                            
                    
                    # Here we generate the declarative from the question.
                    declForm = []
                    if deQuest(newTree) != "INVALID QUESTION":
                        if treeEdit[0] in ["do", "does", "did", "is", "was", "are", "were", "am", "has", "have", "had", 
                                               "can", "may", "might", "must", "will", "wo", "shall", "would", "could", "should"]:
                            for node in deQuest(newTree):
                                if "*" not in node:
                                    declForm.append(node)
                            decl.append(" ".join(declForm))
                            quest.append(" ".join(treeEdit))

    return decl, quest
                    

