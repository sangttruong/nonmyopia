# Amortized Nonmyopic Bayesian Optimization in the Dynamic Cost Settings

This repository implements the paper Amortized Nonmyopic Bayesian Optimization in the Dynamic Cost Settings. We experiment with the following acquisition functions: multi-step H-entropy search (via Monte Carlo integration or Thompson sampling), multi-step trees, simple regret, expected improvement, probability of improvement, upper confidence bound, and knowledge gradient. There are two main experiments. First, we consider multiple synthetic functions with various input dimensions: 2D (Ackley, Alpine, Beale, Branin, EggHolder, Griewank, HolderTable, Levy, SixHumpCamel,  StyblinskiTang, and SynGP), 4D (Powell), 6D (Hartmann), and 8D (Cosine8). In addition, we also consider real-world experiments where we optimize protein sequences to maximize a protein's desirable properties, such as the fluorescent level.

The online experiment is simulated in [main.py](main.py). Starting with initialized data points in the buffer, a Gaussian process regressor is constructed as a surrogate model to guide the decision-making of the actor (in [actor.py](actor.py)) in querying the next observed data point. This data point is chosen to maximize the value of an acquisition function (in [acqfs.py](acqfs.py)). An amortized network (in [amortized_network.py](amortized_network.py)), which is also known as the policy network, can be utilized to reduce the number of parameters when optimizing the acquisition function. The observed data point will be added to the buffer. The buffer is then used to update the surrogate model and guide the actor in collecting more data until the experiment is terminated according to the budget or some information criteria.

Package dependency for this project can be installed via pip:
```bash
pip install -r requirements.txt
```

## Experiment 1: Optimization of Synthetic Functions
Some arguments in this function are the following: `seed` specifies the seed value for random number generation. `task` indicates the type of task on which we run the experiment, which can be `top-k` or `level-set`. `env_name` specifies the name of the synthetic function that is used as the oracle, such as `alpine` or `ackley`. Other arguments include `env_noise`, which adds observation noise to the scores, with possible values like `0.0` or `0.1`. The `env_discretized` argument determines if the embedding space is continuous or discretized. The `algo` argument selects the acquisition function used in the experiment, while `cost_fn` specifies the cost function, with distance metrics like `euclidean` or `manhattan`. `plot` is a true/false boolean that decides whether to plot the loss during training. The `gpu_id` argument allows you to specify the GPU ID if they are visible. Setting `export CUDA_VISIBLE_DEVICES=2,5` will influence this option and assign gpu_id=0 to device 2, gpu_id=1 to device 5. Lastly, the `cont` argument controls whether a started experiment should be resumed( `True` ) or not (`False`). Our example configuration is
```bash
python main.py --seed 2 --task topk --env_name Ackley --env_noise 0.01 --env_discretized False --algo HES-TS-AM-1 --cost_fn euclidean --plot True --gpu_id 0 --cont False
```

Evaluation metrics, such as regret, are computed after the online experiment is complete: we use the same set of arguments to draw its metrics.
```bash
python compute_metrics.py --seed 2 --task topk --env_name Ackley --env_noise 0.01 --env_discretized False --algo HES-TS-AM-1 --cost_fn euclidean --plot True --gpu_id 0 --cont False
```

After that, to visualize those metrics, one can simply run the following script:
```bash
python draw_metrics.py
```

One can scale up the experiment in parallel painlessly with WandB sweep. First, run the below command to get the command to start sweep agent(s). 
```bash
wandb sweep wnb_configs/full.yaml
```
The result will look like "wandb agent your_name/nonmyopia/some_text". To start a single sweep agent.
```bash
CUDA_VISIBLE_DEVICES=0 wandb agent your_name/nonmyopia/some_text &
```
To start more agents, simply rerun the above command for different terminals/servers. You can start as many sweep agents as your server can handle.

## Experiment 2: Optimization of Protein Sequence Property
The oracle in this semi-synthetic experiment is a linear model of the embedding space of the protein sequence. It emulates the outcome measurement from a wet lab experiment. The surrogate model is a linear model trained on the currently available dataset. The policy is a large language model (LLM) that is optimized via PPO to generate sequences with high acquisition values. To select the most suited embedding model, we experiment with various well-known LLMs, such as `meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-v0.1`, `google/gemma-7b`, `zjunlp/llama-molinst-protein-7b`. We first embedded our dataset with various models and then ran the script below. One can compare as many models as they like by specifying a list of `models`.
```bash
python test_oracle_convergence.py \
    --seed 2 \
    --datasets ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --models LLaMa-2 Mistral LLaMa-3 \
    --output_dir results
```

We train linear models (`linear`, `ridge`, or `bayesridge`) using `sklearn`. Adapt the `--dataset` argument with the same personal variables as in `--export_hub_model_id`. Automatically, a huggingface dataset will be created with the same RepoID as your model's.

```bash
python oracle.py \
    --seed 2 \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path bayesridge \
    --dataset_dir data \
    --dataset <HF_USER>/<HF_MODEL> \
    --preprocessing_num_workers 32 \
    --output_dir ckpts/oracle_bayesridge-seed2
```

To run the full pipeline, please use the script in [Pipeline Script](scripts/run_exp.sh).
