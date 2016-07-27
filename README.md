# hieratt
Experimenting with Hierarchical Attention Networks from https://arxiv.org/abs/1606.02393 in Keras

- generate_mref.py generates MNIST reference data set
- model.py implements the Hierarchical Attention Network
- mref_experiment.py is reproducing the MREF experiment from the paper
- vqa_experiment.py is silently failing  

The MREF experiment can be reproduced by running

```bash
  python genera_mref.py mref
```

to create an MREF data set and running 

```bash
  python mref_experiment.py
```
