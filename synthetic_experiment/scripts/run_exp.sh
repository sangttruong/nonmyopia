# Sample run script for running the synthetic experiment
python main.py \
    --seed 42 \
    --env_name SynGP \
    --env_noise 0.0 \
    --algo HES-TS-AM-20 \
    --cost_fn r-spotlight\
    --n_initial_points -1 \
    --n_restarts 64 \
    --hidden_dim 64 \
    --kernel RBF