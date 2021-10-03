# Code modified from https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-grammar
# All code in the public domain

from collections import defaultdict
import random
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Generate random sentences from CFG")
parser.add_argument('--n', type=int, default=10000,
        help='number of sentences to generate')
parser.add_argument('--seed', type=int, default=999,
        help='random seed')
parser.add_argument('--move-first', action='store_true',
        help='use the move first rule')
parser.add_argument('--cn-breakdown', action='store_true',
        help='crain and nakayama breakdown')
args=parser.parse_args()

class CFG(object):
    def __init__(self):
        self.prod = defaultdict(list)

    def __init__(self, cfg, vocab):
        self.prod = defaultdict(list)
        for rule in cfg:
            if rule.strip():
                lhs = rule.split("->")[0].strip()
                rhs = rule.split("->")[1]
                self.add_prod(lhs, rhs)
        for rule in vocab:
            if rule.strip():
                lhs = rule.split("->")[0].strip()
                rhs = rule.split("->")[1]
                self.add_prod(lhs, rhs)

    def add_prod(self, lhs, rhs):
        prods = rhs.split('|')
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))

    def gen_random(self, symbol):
        sentence = ""
        rand_prod = random.choice(self.prod[symbol])
        for sym in rand_prod:
            if sym in self.prod:
                sentence += self.gen_random(sym)
            else:
                sentence += sym + ' ' 
        return sentence 

def move_aux(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    i = decl.index(main_aux_marker)
    decl.pop(i)

    quest = sent.split()
    i = quest.index(main_aux_marker)
    main_aux = quest.pop(i + 1)
    quest.insert(0, main_aux)
    quest.pop(i + 1)

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return main_aux, declarative, question

def move_first(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    decl.pop(decl.index(main_aux_marker))

    quest = sent.split()
    leftmost_aux = quest.pop(quest.index(first_aux(' '.join(quest))))
    quest = [leftmost_aux] + quest
    quest.pop(quest.index(main_aux_marker))
    
    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return leftmost_aux, declarative, question

def prepose_first_and_delete_first(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    decl.pop(decl.index(main_aux_marker))

    quest = sent.split()
    leftmost_aux = quest.pop(quest.index(first_aux(' '.join(quest))))
    quest = [leftmost_aux] + quest
    quest.pop(quest.index(main_aux_marker))

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return leftmost_aux, declarative, question

def prepose_first_and_delete_main(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    decl.pop(decl.index(main_aux_marker))

    quest = sent.split()
    leftmost_aux = quest[quest.index(first_aux(' '.join(quest)))]
    quest = [leftmost_aux] + quest
    quest.pop(quest.index(main_aux_marker) + 1)
    quest.pop(quest.index(main_aux_marker))

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return leftmost_aux, declarative, question

def prepose_first_and_delete_none(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    decl.pop(decl.index(main_aux_marker))

    quest = sent.split()
    leftmost_aux = quest[quest.index(first_aux(' '.join(quest)))]
    quest = [leftmost_aux] + quest
    quest.pop(quest.index(main_aux_marker))

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return leftmost_aux, declarative, question

def prepose_main_and_delete_first(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    i = decl.index(main_aux_marker)
    decl.pop(i)

    quest = sent.split()
    quest.pop(quest.index(first_aux(' '.join(quest))))
    i = quest.index(main_aux_marker)
    main_aux = quest[(i + 1)]
    quest.insert(0, main_aux)
    quest.pop(quest.index(main_aux_marker))

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return main_aux, declarative, question

def prepose_main_and_delete_main(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    i = decl.index(main_aux_marker)
    decl.pop(i)

    quest = sent.split()
    i = quest.index(main_aux_marker)
    main_aux = quest.pop(i + 1)
    quest.insert(0, main_aux)
    quest.pop(quest.index(main_aux_marker))

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return main_aux, declarative, question

def prepose_main_and_delete_none(sent, main_aux_marker="MAIN-AUX"):
    decl = sent.split()
    i = decl.index(main_aux_marker)
    decl.pop(i)

    quest = sent.split()
    i = quest.index(main_aux_marker)
    main_aux = quest[i + 1]
    quest.insert(0, main_aux)
    quest.pop(quest.index(main_aux_marker))

    declarative = ' '.join(decl) + " ."
    question = ' '.join(quest) + " ?"
    return main_aux, declarative, question

def ambiguous(sent, main_aux_marker="MAIN-AUX"):
    words = sent.split()
    i = words.index(main_aux_marker) + 1
    main_aux = words[i]
    return words.index(main_aux) != i

def crain_and_nakayama_breakdown(cfg, vocab, output, n):
    with open(cfg, 'r') as f:
        cfg = f.readlines()
    with open(vocab, 'r') as f:
        vocab = f.readlines()
    grammar = CFG(cfg, vocab)
    data = []
    sentences = []
    i = 0
    while i < n:
        sent = grammar.gen_random('S')
        if not ambiguous(sent):
            main_aux, declarative, question = prepose_first_and_delete_first(sent)
            sentences.append(["prepose_first_and_delete_first", declarative, question])
            main_aux, declarative, question = prepose_first_and_delete_main(sent)
            sentences.append(["prepose_first_and_delete_main", declarative, question])
            main_aux, declarative, question = prepose_first_and_delete_none(sent)
            sentences.append(["prepose_first_and_delete_none", declarative, question])
            main_aux, declarative, question = prepose_main_and_delete_first(sent)
            sentences.append(["prepose_main_and_delete_first", declarative, question])
            main_aux, declarative, question = prepose_main_and_delete_main(sent)
            sentences.append(["prepose_main_and_delete_main", declarative, question])
            main_aux, declarative, question = prepose_main_and_delete_none(sent)
            sentences.append(["prepose_main_and_delete_none", declarative, question])
            i = i + 1
    df = pd.DataFrame(sentences)
    df.loc[:,2:2].to_csv(output, sep="\t", index=False, header=False)
    df.to_csv(output + ".data", sep="\t", index=False, header=False)

def gen(cfg, vocab, output, n):
    with open(cfg, 'r') as f:
        cfg = f.readlines()
    with open(vocab, 'r') as f:
        vocab = f.readlines()
    grammar = CFG(cfg, vocab)
    data = []
    sentences = []
    i = 0
    while i < n:
        sent = grammar.gen_random('S')
        if not ambiguous(sent):
            if args.move_first:
                main_aux, declarative, question = move_first(sent)
            else:
                main_aux, declarative, question = move_aux(sent)
            sentences.append([declarative, question])
            data.append([declarative, question[:-1], first_aux(declarative), main_aux + " ?"])
            i = i + 1
    df = pd.DataFrame(sentences)
    df.to_csv(output, sep="\t", index=False, header=False)
    df2 = pd.DataFrame(data)
    df2.to_csv(output + ".data", sep="\t", index=False, header=False)

def first_aux(sent):
    for word in sent.split():
        if word in ["do","did","can","would","shall","does", "did","can","would","shall",
                "is", "was", "are", "were", "has", "have"]:
            return word


if __name__ == "__main__":
    print("Generating " + str(args.n) + " sentences, with random seed " + str(args.seed))
    random.seed(args.seed)
    if args.cn_breakdown:
        crain_and_nakayama_breakdown(cfg="hierarchical.cfg", vocab="vocab.cfg", output="crain-and-nakayama-breakdown.txt", n=args.n)
    elif args.move_first:
        gen(cfg="linear.cfg", vocab="vocab.cfg", output="linear.mf", n=args.n)
        gen(cfg="hierarchical.cfg", vocab="vocab.cfg", output="hierarchical.mf", n=args.n)
    else: 
        gen(cfg="linear.cfg", vocab="vocab.cfg", output="linear.txt", n=args.n)
        import pdb
        pdb.set_trace()
        gen(cfg="hierarchical.cfg", vocab="vocab.cfg", output="hierarchical.txt", n=args.n)

    print("Finished")
