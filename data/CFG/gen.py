# Code modified from https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-grammar

from collections import defaultdict
import random
import sys
import argparse
import pandas as pd
import math

parser = argparse.ArgumentParser(description="Generate random sentences from CFG")
parser.add_argument('--n', type=int, default=10000,
        help='number of sentences to generate')
parser.add_argument('--seed', type=int, default=999,
        help='random seed')
parser.add_argument('--flip', action='store_true',
        help='generate every sentence with a partner that has the auxs swapped')
parser.add_argument('--cn-breakdown', action='store_true',
        help='generate crain and nakayama breakdown')
parser.add_argument('--mfmm', action='store_true',
        help='generate move first and move main pairs')
parser.add_argument('--slor', action='store_true',
        help='get normalization log likelihoods')
args=parser.parse_args()

aux_list = ["do",
    "did",
    "can",
    "would",
    "shall",
    "does",
    "did",
    "can",
    "would",
    "shall",
    "is",
    "was",
    "are",
    "were",
    "has",
    "have"]

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

def flip_aux(sent): 
    
    # Generating the same sentence with the aux flipped consists of three things steps: 
    # 1. flip the aux 
    # 2. change the form of the verb that follows the aux
    # 3. make each aux agree in number with its subject
    
    sent = sent.split()
    singular_nouns = ["baby","girl","boy","animal","child","person","horse"]
    aux_types = {"do":'basic',
        "did":'basic',
        "can":'basic',
        "would":'basic',
        "shall":'basic',
        "does":'basic',
        "is":'progressive',
        "was":'progressive',
        "are":'progressive',
        "were":'progressive',
        "has":'perfect',
        "have":'perfect'}
    the_main_aux = sent[sent.index('MAIN-AUX') + 1]
    the_main_aux_type = aux_types[the_main_aux]
    the_main_verb = sent[sent.index('MAIN-AUX') + 2]
    the_main_subject = sent[1]
    the_main_subject_number = 'singluar' if the_main_subject in singular_nouns else 'plural'
    the_first_aux = first_aux(' '.join(sent))
    the_first_aux_type = aux_types[the_first_aux]
    the_first_verb = sent[sent.index(the_first_aux) + 1]
    the_embedded_subject = sent[2:][sent[2:].index(the_first_aux) -1]
    the_embedded_subject = the_main_subject if the_embedded_subject == 'who' or the_embedded_subject == 'that' else the_embedded_subject
    the_embedded_subject_number = 'singluar' if the_embedded_subject in singular_nouns else 'plural'
    
    #print(the_main_aux, the_first_aux)
    #print(the_main_verb, the_first_verb)
    #print(the_main_subject, the_embedded_subject)
    
    make_aux_agree = {
        'do': {'singluar':'does', 'plural':'do'},
        'does': {'singluar':'does', 'plural':'do'},
        'did': {'singluar':'did', 'plural':'did'},
        'can': {'singluar':'can', 'plural':'can'},
        'would': {'singluar':'would', 'plural':'would'},
        'shall': {'singluar':'shall', 'plural':'shall'},
        'is': {'singluar':'is', 'plural':'are'},
        'are': {'singluar':'is', 'plural':'are'},
        'was': {'singluar':'was', 'plural':'were'},
        'were': {'singluar':'was', 'plural':'were'},
        'has': {'singluar':'has', 'plural':'have'},
        'have': {'singluar':'has', 'plural':'have'},
    }
    
    new_main_aux = make_aux_agree[the_first_aux][the_main_subject_number]
    new_first_aux = make_aux_agree[the_main_aux][the_embedded_subject_number]
    
    make_verb_agree = {
        "play":{"basic":"play",  "progressive":"playing",  "perfect":"played"},
        "playing":{"basic":"play",  "progressive":"playing",  "perfect":"played"},
        "played":{"basic":"play",  "progressive":"playing",  "perfect":"played"},
        "read":{"basic":"read",  "progressive":"reading",  "perfect":"read"},
        "reading":{"basic":"read",  "progressive":"reading",  "perfect":"read"},
        "draw":{"basic":"draw",  "progressive":"drawing",  "perfect":"drawn"},
        "drawing":{"basic":"draw",  "progressive":"drawing",  "perfect":"drawn"},
        "drawn":{"basic":"draw",  "progressive":"drawing",  "perfect":"drawn"},
        "sit":{"basic":"sit",    "progressive":"sitting",   "perfect":"sat"},
        "sitting":{"basic":"sit",    "progressive":"sitting",   "perfect":"sat"},
        "sat":{"basic":"sit",    "progressive":"sitting",   "perfect":"sat"},
        "fall":{"basic":"fall",  "progressive":"falling",  "perfect":"fallen"},
        "falling":{"basic":"fall",  "progressive":"falling",  "perfect":"fallen"},
        "fallen":{"basic":"fall",  "progressive":"falling",  "perfect":"fallen"},
        "talk":{"basic":"talk",  "progressive":"talking",  "perfect":"talked"},
        "talking":{"basic":"talk",  "progressive":"talking",  "perfect":"talked"},
        "talked":{"basic":"talk",  "progressive":"talking",  "perfect":"talked"},
        "sleep":{"basic":"sleep", "progressive":"sleeping","perfect":"slept"},
        "sleeping":{"basic":"sleep", "progressive":"sleeping","perfect":"slept"},
        "slept":{"basic":"sleep", "progressive":"sleeping","perfect":"slept"},
        "try":{"basic":"try",    "progressive":"trying",    "perfect":"tried"},
        "trying":{"basic":"try",    "progressive":"trying",    "perfect":"tried"},
        "tried":{"basic":"try",    "progressive":"trying",    "perfect":"tried"},
        "work":{"basic":"work",  "progressive":"working",  "perfect":"worked"},
        "working":{"basic":"work",  "progressive":"working",  "perfect":"worked"},
        "worked":{"basic":"work",  "progressive":"working",  "perfect":"worked"},
        "walk":{"basic":"walk",  "progressive":"walking",  "perfect":"walked"},
        "walking":{"basic":"walk",  "progressive":"walking",  "perfect":"walked"},
        "walked":{"basic":"walk",  "progressive":"walking",  "perfect":"walked"},
        "call":{"basic":"call",  "progressive":"calling",  "perfect":"called"},
        "calling":{"basic":"call",  "progressive":"calling",  "perfect":"called"},
        "called":{"basic":"call",  "progressive":"calling",  "perfect":"called"},
        "see":{"basic":"see",    "progressive":"seeing",    "perfect":"seen"},
        "seeing":{"basic":"see",    "progressive":"seeing",    "perfect":"seen"},
        "seen":{"basic":"see",    "progressive":"seeing",    "perfect":"seen"},
        "find":{"basic":"find",  "progressive":"finding",  "perfect":"found"},
        "finding":{"basic":"find",  "progressive":"finding",  "perfect":"found"},
        "found":{"basic":"find",  "progressive":"finding",  "perfect":"found"},
        "help":{"basic":"help",  "progressive":"helping",  "perfect":"helped"},
        "helping":{"basic":"help",  "progressive":"helping",  "perfect":"helped"},
        "helped":{"basic":"help",  "progressive":"helping",  "perfect":"helped"},
        "feed":{"basic":"feed",  "progressive":"feeding",  "perfect":"fed"},
        "feeding":{"basic":"feed",  "progressive":"feeding",  "perfect":"fed"},
        "fed":{"basic":"feed",  "progressive":"feeding",  "perfect":"fed"},
        "know":{"basic":"know",  "progressive":"knowing",  "perfect":"known"},
        "knowing":{"basic":"know",  "progressive":"knowing",  "perfect":"known"},
        "known":{"basic":"know",  "progressive":"knowing",  "perfect":"known"},
        "pick":{"basic":"pick",  "progressive":"picking",  "perfect":"picked"},
        "picking":{"basic":"pick",  "progressive":"picking",  "perfect":"picked"},
        "picked":{"basic":"pick",  "progressive":"picking",  "perfect":"picked"},
        "visit":{"basic":"visit", "progressive":"visiting","perfect":"visited"},
        "visiting":{"basic":"visit", "progressive":"visiting","perfect":"visited"},
        "visited":{"basic":"visit", "progressive":"visiting","perfect":"visited"},
        "watch":{"basic":"watch", "progressive":"watching","perfect":"watched"},
        "watching":{"basic":"watch", "progressive":"watching","perfect":"watched"},
        "watched":{"basic":"watch", "progressive":"watching","perfect":"watched"},
        "reach":{"basic":"reach", "progressive":"reaching","perfect":"reached"},
        "reaching":{"basic":"reach", "progressive":"reaching","perfect":"reached"},
        "reached":{"basic":"reach", "progressive":"reaching","perfect":"reached"}
    }
    #print(the_main_aux_type)
    new_main_verb = make_verb_agree[the_main_verb][the_first_aux_type]
    new_first_verb = make_verb_agree[the_first_verb][the_main_aux_type]
    
    #print(the_main_verb, the_first_verb)
    #print(new_main_verb, new_first_verb)
    
    sent[sent.index(the_first_aux)] = new_first_aux
    sent[sent.index(the_first_verb)] = new_first_verb
    sent[sent.index('MAIN-AUX')+1] = new_main_aux
    sent[sent.index('MAIN-AUX')+2] = new_main_verb
    return(' '.join(sent))

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

def first_main_pairs(cfg, vocab, output, n):
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
        if ambiguous(sent):
            main_aux, declarative, question = prepose_first_and_delete_first(sent)
            sentences.append(["prepose_first_and_delete_first", declarative, question])
            main_aux, declarative, question = prepose_main_and_delete_main(sent)
            sentences.append(["prepose_main_and_delete_main", declarative, question])
            i = i + 1
    df = pd.DataFrame(sentences)
    df.loc[:,2:2].to_csv(output, sep="\t", index=False, header=False)
    df.loc[:,[0,2]].to_csv(output + ".data", sep="\t", index=False, header=False)

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
        decl = grammar.gen_random('S')
        if not ambiguous(decl):
            for sent in [decl, flip_aux(decl)]:
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
                #for aux in aux_list:
                #    if aux not in sent.split()[:sent.split().index(main_aux) + 1]:
                #        main_aux, declarative, question = prepose_main_and_delete_none(sent)
                #        sentences.append(["prepose_other_and_delete_none", declarative, aux + ' ' + ' '.join(question.split()[1:])])
                i = i + 1
    df = pd.DataFrame(sentences)
    df.loc[:,2:2].to_csv(output, sep="\t", index=False, header=False)
    df.loc[:,[0,2]].to_csv(output + ".data", sep="\t", index=False, header=False)

def gen(cfg, vocab, output, n, flip):
    with open(cfg, 'r') as f:
        cfg = f.readlines()
    with open(vocab, 'r') as f:
        vocab = f.readlines()
    grammar = CFG(cfg, vocab)
    data = []
    sentences = []
    i = 0
    while i < n:
        decl = grammar.gen_random('S')
        if not ambiguous(decl):
            for sent in [decl, flip_aux(decl)] if flip else [decl]:
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
        if word in aux_list: #aux_list defined at top of file
            return word

aux_probs = {"do":0.014347633100491171,
    "did":0.004617462791847254,
    "can":0.006787976939183948,
    "would":0.0014369039706592958,
    "shall":0.0001917337638413958,
    "does":0.0026889011491529,
    "did":0.004617462791847254,
    "can":0.006787976939183948,
    "would":0.0014369039706592958,
    "shall":0.0001917337638413958,
    "is":0.009950253361647139,
    "was":0.0032421172776536324,
    "are":0.005021874080094494,
    "were":0.0010149045519932906,
    "has":0.0009420063798628866,
    "have":0.005526028581637621}

vocab_prob = { "the":0.02402376619265112270,
        "some":0.00236178506564075404,
        "those":0.00119425719771095113,
        "this":0.00766969768780883382,
        "baby":0.00173162086655792883,
        "girl":0.00052394114826423671,
        "boy":0.00080049135682243582,
        "animal":0.00012716681138303800,
        "child":0.00003934187067355134,
        "person":0.00010587591666558670,
        "horse":0.00020770193487948429,
        "babies":0.00013179526675639698,
        "girls":0.00011073579480761362,
        "boys":0.00014857341748482328,
        "animals":0.00039839429626187429,
        "children":0.00011432284772196684,
        "people":0.00047152389116094620,
        "horses":0.00004674739927092571,
        "play":0.00160595830317123247,
        "read":0.00067980438296210040,
        "draw":0.00017703841803098103,
        "sit":0.00077943188487365246,
        "fall":0.00022586862221991828,
        "talk":0.00045393576074218207,
        "sleep":0.00021267752440584517,
        "try":0.00059128517394660982,
        "work":0.00031600779061608441,
        "walk":0.00020446201611813299,
        "playing":0.00034562990500558194,
        "reading":0.00008840349763115654,
        "drawing":0.00004802022449859943,
        "sitting":0.00016280591775790217,
        "falling":0.00004987160664794303,
        "talking":0.00023616693542564200,
        "sleeping":0.00009280053023584757,
        "trying":0.00027307886702817989,
        "working":0.00008655211548181295,
        "walking":0.00010633876220292259,
        "played":0.00011663707540864633,
        "read":0.00067980438296210040,
        "drawn":0.00000694268306003847,
        "sat":0.00003610195191220005,
        "fallen":0.00000347134153001924,
        "talked":0.00003043209407983530,
        "slept":0.00001029831320572373,
        "tried":0.00006884827367871484,
        "worked":0.00001781955318743208,
        "walked":0.00003043209407983530,
        "call":0.00028476571684591133,
        "see":0.00446032673192171651,
        "find":0.00048760777358336868,
        "help":0.00058306966565889766,
        "feed":0.00014938339717516112,
        "know":0.00435792215678614848,
        "pick":0.00031392498569807290,
        "visit":0.00007382386320507575,
        "watch":0.00047707803760897700,
        "reach":0.00005241725710329046,
        "calling":0.00004223465528190070,
        "seeing":0.00002534079316914042,
        "finding":0.00000960404489971989,
        "helping":0.00002672932978114812,
        "feeding":0.00003054780546416928,
        "knowing":0.00000312420737701731,
        "picking":0.00004107754143856096,
        "visiting":0.00000578556921669873,
        "watching":0.00006653404599203536,
        "reaching":0.00000821550828771219,
        "called":0.00039735289380286854,
        "seen":0.00021510746347685864,
        "found":0.00021452890655518876,
        "helped":0.00002256371994512503,
        "fed":0.00001295967504540515,
        "known":0.00000890977659371604,
        "picked":0.00006074847677533663,
        "visited":0.00000601699198536668,
        "watched":0.00002325798825112888,
        "reached":0.00000659554890703655,
        "do":0.01434763310049117120,
        "shall":0.00019173376384139579,
        "does":0.00268890114915290015,
        "did":0.00461746279184725369,
        "can":0.00678797693918394788,
        "would":0.00143690397065929579,
        "is":0.00995025336164713850,
        "was":0.00324211727765363238,
        "are":0.00502187408009449436,
        "were":0.00101490455199329055,
        "has":0.00094200637986288660,
        "have":0.00552602858163762142,
        "by":0.00041760238606131409,
        "behind":0.00015864030792187909,
        "who":0.00217410120025104741,
        "that":0.01785021670428058038,
        "?":0.05128849254911253819}

# prepose_first_and_delete_first
# prepose_first_and_delete_main
# prepose_first_and_delete_none
# prepose_main_and_delete_first
# prepose_main_and_delete_main
# prepose_main_and_delete_none
def slor(datafile):
    with open (datafile) as f:
        sents = [x.split()[1:] for x in f.readlines()]
    slors = []
    for sent in sents:
        slors.append(sum([math.log(vocab_prob[word]) for word in sent])/len(sent))
    with open('slors.txt','w') as f:
        f.write('\n'.join([str(s) for s  in slors]))

if __name__ == "__main__":
    print("Generating " + str(args.n) + " sentences, with random seed " + str(args.seed))
    random.seed(args.seed)
    if args.cn_breakdown:
        crain_and_nakayama_breakdown(cfg="hierarchical.cfg", vocab="vocab.cfg", output="crain-and-nakayama-breakdown.txt", n=args.n)
    elif args.slor:
        slor('crain-and-nakayama-breakdown.txt.data')
    elif args.mfmm:
        first_main_pairs(cfg="hierarchical.cfg", vocab="vocab.cfg", output="first_main_pairs.txt", n=args.n)
    else: 
        gen(cfg="linear.cfg", vocab="vocab.cfg", output="linear.txt", n=args.n, flip=args.flip)
        gen(cfg="hierarchical.cfg", vocab="vocab.cfg", output="hierarchical.txt", n=args.n, flip=args.flip)

    print("Finished")
