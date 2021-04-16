import os

from CHILDES_xml_Processing import process_childes_xml
from CHILDES_txt_Processing import clean_and_unk
from CHILDES_Treebank_Processing import process_childes_treebank
from CHILDES_Treebank_txt_Processing import split_treebank

train_raw, valid_raw, test_raw, excluded_raw = process_childes_xml("./", "childes-xml")
train, valid, test, excluded, vocab = clean_and_unk(train_raw, valid_raw, test_raw, excluded_raw)
if not os.path.exists('pretraining'):
    os.mkdir('pretraining')
with open ('pretraining/train.txt', 'w') as f:
    f.write("\n".join([' '.join(s) for s in train]))
with open ('pretraining/valid.txt', 'w') as f:
    f.write("\n".join([' '.join(s) for s in valid]))
with open ('pretraining/test.txt', 'w') as f:
    f.write("\n".join([' '.join(s) for s in test]))
with open ('pretraining/excluded.txt', 'w') as f:
    f.write("\n".join([' '.join(s) for s in excluded]))
with open ('pretraining/vocab.txt', 'w') as f:
    f.write("\n".join(vocab))

decl, quest = process_childes_treebank("childes-treebank")
with open ('pretraining/excluded.txt') as f:
    excluded = f.read()
finetuning_train, finetuning_valid, finetuning_test = split_treebank(excluded, decl, quest)
if not os.path.exists('finetuning'):
    os.mkdir('finetuning')
with open ('finetuning/train.txt', 'w') as f:
    f.write("".join(finetuning_train))
with open ('finetuning/valid.txt', 'w') as f:
    f.write("".join(finetuning_valid))
with open ('finetuning/test.txt', 'w') as f:
    f.write("".join(finetuning_test))
