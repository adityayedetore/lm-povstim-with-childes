# Generating Targeted Evaluation Set

The evaluation datasets can be downloaded from [here](http://www.adityayedetore.com/childes-project/). 

Alternatively, to generate the data, run `python gen.py --n 10000 --flip`, which will produce `hierarchical.txt`, `hierarchical.txt.data`, `linear.txt`, and `linear.txt`  

The `.txt` and `.txt.data` are very similar, except the `.txt.data` files are formatted to make model evaluation with `/src/eval.py` simple.
