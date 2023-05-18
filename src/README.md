This directory contains the scripts we used to run our experiments.

For all the commands above below, to run with cuda append `--cuda` to the `python main.py ...` or `python eval.py ...` call.

## Training 

Before training the models, download the `pretraining.zip` from [here](http://www.adityayedetore.com/childes-project/), unzip it, and put it in `/data/CHILDES`.

To train a transformer model, use 

`python main.py --data ../data/CHILDES/pretraining --nlayers 4 --bptt 500 --nhid 800 --emsize 800 --lr 5.0 --batch_size 10 --dropout 0.2 --nhead 4 --seed 1001 --model Transformer --save models/transformer.pt --log models/transformer.log`

To train a lstm model, use

`python main.py --data ../data/CHILDES/pretraining --nlayers 2 --nhid 800 --emsize 800 --lr 10.0 --batch_size 20 --dropout 0.4 --seed 1001 --model LSTM --save models/lstm.pt --log models/lstm.log`

## Finetuning

Before training the models, train the models with the above commands, then download the `finetuning.zip` from [here](http://www.adityayedetore.com/childes-project/), unzip it, and put it in `/data/CHILDES`.

To finetune a transformer model, use 

`python main.py --data ../data/CHILDES/finetuning --nlayers 4 --bptt 500 --nhid 800 --emsize 800 --lr 5.0 --batch_size 10 --dropout 0.2 --nhead 4 --seed 1001 --model Transformer --load models/transformer.pt --save models/transformer-finetuned.pt --log models/transformer-finetuned.log --finetune`

To finetune a lstm model, use 

`python main.py --data ../data/CHILDES/finetuning --nlayers 2 --nhid 800 --emsize 800 --lr 10.0 --batch_size 20 --dropout 0.4 --seed 1001 --model LSTM --load models/lstm.pt --save models/lstm-finetuned.pt --log models/lstm-finetuned.log --finetune`

### Evaluating

To evaluate the models, first train and finetune models using the above commands, then call the following: 

`python eval.py --data ../data/CFG/linear.txt.data --finetuning_data ../data/CHILDES/finetuning/ --model models/lstm-finetuned.pt --rnn --results results/lstm-linear-results.txt`

`python eval.py --data ../data/CFG/hierarchical.txt.data --finetuning_data ../data/CHILDES/finetuning/ --model models/lstm-finetuned.pt --rnn --results results/lstm-hierarchical-results.txt`

`python eval.py --data ../data/CFG/linear.txt.data --finetuning_data ../data/CHILDES/finetuning/ --model models/transformer-finetuned.pt --results results/transformer-linear-results.txt`

`python eval.py --data ../data/CFG/hierarchical.txt.data --finetuning_data ../data/CHILDES/finetuning/ --model models/transformer-finetuned.pt --results results/transformer-hierarchical-results.txt`

Calling `eval.py` will print some summary information, such as the proportion of model responses consistent with `MOVE-FIRST` or `MOVE-MAIN`, along with generating the files containing the actual predictions of the models. 