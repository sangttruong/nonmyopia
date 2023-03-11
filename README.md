# h-entropy-search

Bayesian optimization with H-Entropy Search. This repo includes code for both the myopic (one-step) and nonmyopic (multi-step) versions.


## Installation

To install dependencies, `cd` into this repo directory and run:
```bash
$ pip install -r requirements.txt
```


## Running Examples

First make sure this repo directory is on the PYTHONPATH, e.g. by running:
```bash
$ source shell/add_pwd_to_pythonpath.sh
```

For example, if we want to run an optimization procedure on synthetic function with H-entropy search:

Using Monte Carlo one-step with 0.1 spotlight cost: 
```bash
$ python examples/opt_synthfunc.py --algo 'hes_mc' --r 0.1 --lookahead_step 1
```

By default, we use 128 restarts for the acquisition optimization. When using Variational version, we typically need to reduce the number of restarts to a smaller number than default to speed up the computation:
```bash
$ python examples/opt_synthfunc.py --algo 'hes_vi' --r 0.1 --lookahead_step 1 --n_restarts 32
```

For the semisynthetic setting using synthetic graph, following these steps: 
* Download the ChEMBL dataset: 
```bash
wget https://github.com/kevinid/molecule_generator/releases/download/1.0/datasets.tar.gz
```
* Unzip it under the following directory `h-entropy-search/examples/rexgen_direct/src/datasets`. 
* Download pretrained reaction prediction models and unzip under their corresponding folders: `h-entropy-search/examples/rexgen_direct/src/core_wln_global/model-300-3-direct` and 
`h-entropy-search/examples/rexgen_direct/src/rank_diff_wln/model-core16-500-3-max150-direct-useScores`
```bash
wget https://github.com/connorcoley/rexgen_direct/tree/master/rexgen_direct/core_wln_global/model-300-3-direct

wget https://github.com/connorcoley/rexgen_direct/tree/master/rexgen_direct/rank_diff_wln/model-core16-500-3-max150-direct-useScores
``` 
* Running the following command: 
```bash
cd /h-entropy-search/examples/rexgen_direct
source shell/add_pwd_to_pythonpath.sh
python experiments/get_synthetic_graph.py
```
Some options for generation process of the synthetic graph are the number of edges or the source data (ZINC vs ChEMBL). 