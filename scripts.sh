#!/bin/bash
# SynGP - Non-myopic
python _0_main.py --exp_id 1 \
                  --gpu_id 3 \
                  --algo HES \
                  --env_name SynGP \
                  --seeds 2 3 5  \
                  --n_iterations 12 \
                  --lookahead_steps 12 \
                  --bounds -1 1

# SynGP - Myopic
python _0_main.py --exp_id 2 \
                  --gpu_id 3 \
                  --algo HES \
                  --env_name SynGP \
                  --seeds 2 3 5  \
                  --n_iterations 12 \
                  --lookahead_steps 1 \
                  --bounds -1 1