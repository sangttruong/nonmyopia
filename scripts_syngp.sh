#!/bin/bash

# Declear the intended GPU id
gpu_id=(0 1 2 3 4 5 6 7 8 9)
# Declare an array contraining all the seeds
seeds=(1 2 3 4 5 6 7 8 9 10)
# Declear an array containing all the algorithms
algos=(qMSL qSR qEI qPI qUCB qKG)
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
    for algo in "${algos[@]}"
    do
        # Print the current experiment
        echo "Running experiment for $env and $algo"

        python _0_main.py --gpu_id ${gpu_id[$idx]} \
                          --algo $algo \
                          --env_name $env \
                          --seeds ${seeds[@]} \
                          --n_iterations ${env_iterations[$env]} \
                          --lookahead_steps ${algo_lookahead[$env,$algo]} \
                          --bounds ${env_lower_bounds[$env]} ${env_upper_bounds[$env]} &

        if [ "$algo" == "HES" ]
        then
            idx=$((idx+1))
            python _0_main.py --gpu_id ${gpu_id[$idx]} \
                              --algo $algo \
                              --env_name $env \
                              --seeds ${seeds[@]} \
                              --n_iterations ${env_iterations[$env]} \
                              --lookahead_steps 1 \
                              --bounds ${env_lower_bounds[$env]} ${env_upper_bounds[$env]} & 
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
#                          --algo $algo \
#                          --env_name $env \
#                          --seeds 0 \
#                          --n_iterations ${env_iterations[$env]} \
#                          --lookahead_steps ${algo_lookahead[$env,$algo]} \
#                          --bounds ${env_lower_bounds[$env]} ${env_upper_bounds[$env]} \
#                          --discetized

#     if [ "$algo" == "HES" ]
#     then
#         exp_id=$((exp_id+1))
#         python _0_main.py            --gpu_id $gpu_id \
#                              --algo $algo \
#                              --env_name $env \
#                              --seeds 0 \
#                              --n_iterations ${env_iterations[$env]} \
#                              --lookahead_steps 1 \
#                              --bounds ${env_lower_bounds[$env]} ${env_upper_bounds[$env]} \
#                              --discetized
#     fi
# done



# #=========================================================
# #            SYNTHETIC GP DISCRETE
# #=========================================================

# python _0_main_dc.py 
#                   --gpu_id 4 \
#                   --algo HES \
#                   --env_name SynGP \
#                   --seeds 0 \
#                   --n_iterations 18 \
#                   --lookahead_steps 15 \
#                   --bounds -1 1

# python _0_main_dc.py 
#                   --gpu_id 3 \
#                   --algo HES \
#                   --env_name SynGP \
#                   --seeds 0 \
#                   --n_iterations 18 \
#                   --lookahead_steps 1 \
#                   --bounds -1 1

# python _0_main_dc.py 
#                   --gpu_id 3 \
#                   --algo qMSL \
#                   --env_name SynGP \
#                   --seeds 0 \
#                   --n_iterations 18 \
#                   --lookahead_steps 3 \
#                   --bounds -1 1

# python _0_main_dc.py 
#                   --gpu_id 3 \
#                   --algo qEI \
#                   --env_name SynGP \
#                   --seeds 0 \
#                   --n_iterations 18 \
#                   --lookahead_steps 0 \
#                   --bounds -1 1



# #=========================================================
# #            SYNTHETIC GP 
# #=========================================================

# # SynGP - HES - Non-myopic
# python _0_main.py --gpu_id 9 \
#                   --algo HES \
#                   --env_name SynGP \
#                   --seeds 10 \
#                   --n_iterations 30 \
#                   --lookahead_steps 15 \
#                   --bounds -1 1 \
#                   --test_only \
#                   --continue_once "results/exp_HES_SynGP_15" &

# # SynGP - HES - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algo HES \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 1 \
#                   --bounds -1 1

# # SynGP - qMSL - Non-myopic
# python _0_main.py --gpu_id 4 \
#                   --algo qMSL \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 3 \
#                   --bounds -1 1

# # SynGP - qSR - Myopic
# python _0_main.py --gpu_id 5 \
#                   --algo qSR \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 0 \
#                   --bounds -1 1

# # SynGP - qEI - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algo qEI \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 0 \
#                   --bounds -1 1

# # SynGP - qPI - Myopic
# python _0_main.py --gpu_id 8 \
#                   --algo qPI \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 0 \
#                   --bounds -1 1

# # SynGP - qUCB - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algo qUCB \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 0 \
#                   --bounds -1 1

# # SynGP - qKG - Myopic
# python _0_main.py --gpu_id 8 \
#                   --algo qKG \
#                   --env_name SynGP \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 18 \
#                   --lookahead_steps 0 \
#                   --bounds -1 1



# #=========================================================
# #            HOLDER TABLE
# #=========================================================


# # HolderTable - HES - Non-myopic
# python _0_main.py --gpu_id 6 \
#                   --algo HES \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 12 \
#                   --bounds 1 10

# # HolderTable - HES - Myopic
# python _0_main.py --gpu_id 6 \
#                   --algo HES \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 1 \
#                   --bounds 1 10

# # HolderTable - qMSL - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algo qMSL \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 3 \
#                   --bounds 1 10

# # HolderTable - qSR - Myopic
# python _0_main.py --gpu_id 0 \
#                   --algo qSR \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 0 \
#                   --bounds 1 10

# # HolderTable - qEI - Myopic
# python _0_main.py --gpu_id 0 \
#                   --algo qEI \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 0 \
#                   --bounds 1 10

# # HolderTable - qPI - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algo qPI \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 0 \
#                   --bounds 1 10

# # HolderTable - qUCB - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algo qUCB \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 0 \
#                   --bounds 1 10

# # HolderTable - qKG - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algo qKG \
#                   --env_name HolderTable \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 21 \
#                   --lookahead_steps 0 \
#                   --bounds 1 10






# #=========================================================
# #            EGG HOLDER 
# #=========================================================


# # EggHolder - HES - Non-myopic
# python _0_main.py --gpu_id 7 \
#                   --algo HES \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 10 \
#                   --bounds -512 512

# # EggHolder - HES - Myopic
# python _0_main.py --gpu_id 2 \
#                   --algo HES \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 1 \
#                   --bounds -512 512

# # EggHolder - qMSL - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algo qMSL \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 3 \
#                   --bounds -512 512

# # EggHolder - qSR - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algo qSR \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 0 \
#                   --bounds -512 512

# # EggHolder - qEI - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algo qEI \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 0 \
#                   --bounds -512 512

# # EggHolder - qPI - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algo qPI \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 0 \
#                   --bounds -512 512

# # EggHolder - qUCB - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algo qUCB \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 0 \
#                   --bounds -512 512

# # EggHolder - qKG - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algo qKG \
#                   --env_name EggHolder \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 16 \
#                   --lookahead_steps 0 \
#                   --bounds -512 512




# #=========================================================
# #            ALPINE
# #=========================================================


# # Alpine - HES - Non-myopic
# python _0_main.py --gpu_id 3 \
#                   --algo HES \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 11 \
#                   --bounds 0 10

# # Alpine - HES - Myopic
# python _0_main.py --gpu_id 0 \
#                   --algo HES \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 1 \
#                   --bounds 0 10

# # Alpine - qMSL - Myopic
# python _0_main.py --gpu_id 9 \
#                   --algo qMSL \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 3 \
#                   --bounds 0 10

# # Alpine - qSR - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algo qSR \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 0 \
#                   --bounds 0 10

# # Alpine - qEI - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algo qEI \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 0 \
#                   --bounds 0 10

# # Alpine - qPI - Myopic
# python _0_main.py --gpu_id 3 \
#                   --algo qPI \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 0 \
#                   --bounds 0 10

# # Alpine - qUCB - Myopic
# python _0_main.py --gpu_id 4 \
#                   --algo qUCB \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 0 \
#                   --bounds 0 10

# # Alpine - qKG - Myopic
# python _0_main.py --gpu_id 7 \
#                   --algo qKG \
#                   --env_name Alpine \
#                   --seeds 1 2 3 4 5 6 7 8 9 10 \
#                   --n_iterations 17 \
#                   --lookahead_steps 0 \
#                   --bounds 0 10
