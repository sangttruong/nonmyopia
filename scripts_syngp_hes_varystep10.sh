#!/bin/bash

# Declear the intended GPU id
gpu_id=(1 2 3 4 5 6 7 8)
# Declare an array contraining all the seeds
seeds=(1 2 3 4 5 6 7 8)
# Declear an array containing all the algorithms
algos=(HES)
# Declare an array containing all environments
envs=(SynGP)
# Declare a hash map containing all the bounds for each environment
declare -A env_lower_bounds
declare -A env_upper_bounds
env_lower_bounds=(["SynGP"]=-1 ["HolderTable"]=1 ["EggHolder"]=-512 ["Alpine"]=0)
env_upper_bounds=(["SynGP"]=1 ["HolderTable"]=10 ["EggHolder"]=512 ["Alpine"]=10)
# Declare a hash map containing all the iterations for each environment
declare -A env_iterations
env_iterations=(["SynGP"]=30 ["HolderTable"]=30 ["EggHolder"]=16 ["Alpine"]=30)
# Declare a hash map containing all the lookahead steps for each algorithm in each environment
declare -A algo_lookahead
algo_lookahead=( \
        ["SynGP,HES"]=15 ["SynGP,qMSL"]=3 ["SynGP,qSR"]=0 ["SynGP,qEI"]=0 \
        ["SynGP,qPI"]=0 ["SynGP,qUCB"]=0 ["SynGP,qKG"]=0 \
        ["HolderTable,HES"]=15 ["HolderTable,qMSL"]=3 ["HolderTable,qSR"]=0 ["HolderTable,qEI"]=0 \
        ["HolderTable,qPI"]=0 ["HolderTable,qUCB"]=0 ["HolderTable,qKG"]=0 \
        ["EggHolder,HES"]=10 ["EggHolder,qMSL"]=3 ["EggHolder,qSR"]=0 ["EggHolder,qEI"]=0 \
        ["EggHolder,qPI"]=0 ["EggHolder,qUCB"]=0 ["EggHolder,qKG"]=0 \
        ["Alpine,HES"]=15 ["Alpine,qMSL"]=3 ["Alpine,qSR"]=0 ["Alpine,qEI"]=0 \
        ["Alpine,qPI"]=0 ["Alpine,qUCB"]=0 ["Alpine,qKG"]=0 \
)
        

# Write for loop to run all the experiments
# When using HES agorithm, run the non-myopic and myopic versions separately
# The non-myopic version has lookahead steps as defined in algo_lookahead
# The myopic version has lookahead steps as 1
# The non-myopic version has experiment id as defined in env_algo_exp_ids
# The myopic version has experiment id as defined in env_algo_exp_ids + 1

# Declare a variable to count the experiment id
for env in "${envs[@]}"
do
    idx=0
    for seed in "${seeds[@]}"
    do
        for algo in "${algos[@]}"
        do
            # Print the current experiment
            echo "Running experiment for $env and $algo"

            python _0_main.py --gpu_id ${gpu_id[$idx]} \
                            --algo $algo \
                            --env_name $env \
                            --seeds ${seeds[$idx]} \
                            --n_iterations ${env_iterations[$env]} \
                            --lookahead_steps 10 \
                            --bounds ${env_lower_bounds[$env]} ${env_upper_bounds[$env]} &
        done
        idx=$((idx+1))
    done
    wait
done