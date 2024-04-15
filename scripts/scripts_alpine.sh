#!/bin/bash

# Declear the intended GPU id
gpu_id=(0 1 2 3 4 5 6 7 8 9)
# Declare an array contraining all the seeds
seeds=(1 2 3 4 5 6 7 8 9 10)
# Declear an array containing all the algorithms
algos=(qMSL qSR qEI qPI qUCB qKG)
# Declare an array containing all environments
envs=(Alpine)
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
    for algo in "${algos[@]}"
    do
        # Print the current experiment
        echo "Running experiment for $env and $algo"

        python _0_main.py --gpu_id ${gpu_id[$idx]} \
                          --algos $algo \
                          --env_names $env \
                          --seeds ${seeds[@]} \
                          --algo_n_iterations ${env_iterations[$env]} \
                          --algo_lookahead_steps ${algo_lookahead[$env,$algo]} \

        if [ "$algo" == "HES" ]
        then
            idx=$((idx+1))
            python _0_main.py --gpu_id ${gpu_id[$idx]} \
                              --algos $algo \
                              --env_names $env \
                              --seeds ${seeds[@]} \
                              --algo_n_iterations ${env_iterations[$env]} \
                              --algo_lookahead_steps 1 \
        fi
        idx=$((idx+1))
    done
    wait
done

# Write for loop to run all the experiments in discrete setting with seed 0
# When using HES agorithm, run the non-myopic and myopic versions separately
# The non-myopic version has lookahead steps as defined in algo_lookahead
# The myopic version has lookahead steps as 1
# The non-myopic version has experiment id as defined in env_algo_exp_ids
# The myopic version has experiment id as defined in env_algo_exp_ids + 1
# exp_id=90
# env="SynGP"
# for algo in "${algos[@]}"
# do
#     # Print the current experiment
#     echo "Running discrete experiment for $env and $algo"
#     exp_id=$((exp_id+1))
#     python _0_main.py        --gpu_id $gpu_id \
#                          --algos $algo \
#                          --env_names $env \
#                          --seeds 0 \
#                          --algo_n_iterations ${env_iterations[$env]} \
#                          --algo_lookahead_steps ${algo_lookahead[$env,$algo]} \
#                          --env_discretized

#     if [ "$algo" == "HES" ]
#     then
#         exp_id=$((exp_id+1))
#         python _0_main.py            --gpu_id $gpu_id \
#                              --algos $algo \
#                              --env_names $env \
#                              --seeds 0 \
#                              --algo_n_iterations ${env_iterations[$env]} \
#                              --algo_lookahead_steps 1 \
#                              --env_discretized
#     fi
# done



# #=========================================================
# #            SYNTHETIC GP DISCRETE
# #=========================================================

# python _0_main_dc.py 
#                   --gpu_id 4 \
#                   --algos HES \
#                   --env_names SynGP \
#                   --seeds 0 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 15 \

# python _0_main_dc.py 
#                   --gpu_id 3 \
#                   --algos HES \
#                   --env_names SynGP \
#                   --seeds 0 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 1 \

# python _0_main_dc.py 
#                   --gpu_id 3 \
#                   --algos qMSL \
#                   --env_names SynGP \
#                   --seeds 0 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 3 \

# python _0_main_dc.py 
#                   --gpu_id 3 \
#                   --algos qEI \
#                   --env_names SynGP \
#                   --seeds 0 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 0 \



# #=========================================================
# #            SYNTHETIC GP 
# #=========================================================

# # SynGP - HES - Non-myopic
# python _0_main.py --gpu_id 9 \
#                   --algos HES \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 15 \

# # SynGP - HES - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algos HES \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 1 \

# # SynGP - qMSL - Non-myopic
# python _0_main.py --gpu_id 4 \
#                   --algos qMSL \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 3 \

# # SynGP - qSR - Myopic
# python _0_main.py --gpu_id 5 \
#                   --algos qSR \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 0 \

# # SynGP - qEI - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algos qEI \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 0 \

# # SynGP - qPI - Myopic
# python _0_main.py --gpu_id 8 \
#                   --algos qPI \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 0 \

# # SynGP - qUCB - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algos qUCB \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 0 \

# # SynGP - qKG - Myopic
# python _0_main.py --gpu_id 8 \
#                   --algos qKG \
#                   --env_names SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 18 \
#                   --algo_lookahead_steps 0 \



# #=========================================================
# #            HOLDER TABLE
# #=========================================================


# # HolderTable - HES - Non-myopic
# python _0_main.py --gpu_id 6 \
#                   --algos HES \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 12 \

# # HolderTable - HES - Myopic
# python _0_main.py --gpu_id 6 \
#                   --algos HES \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 1 \

# # HolderTable - qMSL - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algos qMSL \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 3 \

# # HolderTable - qSR - Myopic
# python _0_main.py --gpu_id 0 \
#                   --algos qSR \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 0 \

# # HolderTable - qEI - Myopic
# python _0_main.py --gpu_id 0 \
#                   --algos qEI \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 0 \

# # HolderTable - qPI - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algos qPI \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 0 \

# # HolderTable - qUCB - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algos qUCB \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 0 \

# # HolderTable - qKG - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algos qKG \
#                   --env_names HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 21 \
#                   --algo_lookahead_steps 0 \






# #=========================================================
# #            EGG HOLDER 
# #=========================================================


# # EggHolder - HES - Non-myopic
# python _0_main.py --gpu_id 7 \
#                   --algos HES \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 10 \

# # EggHolder - HES - Myopic
# python _0_main.py --gpu_id 2 \
#                   --algos HES \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 1 \

# # EggHolder - qMSL - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algos qMSL \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 3 \

# # EggHolder - qSR - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algos qSR \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 0 \

# # EggHolder - qEI - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algos qEI \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 0 \

# # EggHolder - qPI - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algos qPI \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 0 \

# # EggHolder - qUCB - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algos qUCB \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 0 \

# # EggHolder - qKG - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algos qKG \
#                   --env_names EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 16 \
#                   --algo_lookahead_steps 0 \




# #=========================================================
# #            ALPINE
# #=========================================================


# # Alpine - HES - Non-myopic
# python _0_main.py --gpu_id 3 \
#                   --algos HES \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 11 \

# # Alpine - HES - Myopic
# python _0_main.py --gpu_id 0 \
#                   --algos HES \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 1 \

# # Alpine - qMSL - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algos qMSL \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 3 \

# # Alpine - qSR - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algos qSR \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 0 \

# # Alpine - qEI - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algos qEI \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 0 \

# # Alpine - qPI - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algos qPI \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 0 \

# # Alpine - qUCB - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algos qUCB \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 0 \

# # Alpine - qKG - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algos qKG \
#                   --env_names Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --algo_n_iterations 17 \
#                   --algo_lookahead_steps 0 \
