#!/bin/bash

# Declear the intended GPU id
gpu_id=(0 1 2 3 4 5 6 7 8 9)

# Declare an array contraining all the seeds
seeds=(2 3 5 7 11)

# Declear an array containing all the algorithms
algos=(HES qMSL qSR qEI qPI qUCB qKG)

# Declare an array containing all environments
envs=(SynGP)

# Declare a hash map containing all the iterations for each environment
declare -A env_iterations
env_iterations=( \
        ["Ackley"]=120 \
        ["Alpine"]=150 \
        ["Beale"]=75 \
        ["Branin"]=20 \
        ["Cosine8"]=200 \
        ["EggHolder"]=150 \
        ["Griewank"]=20 \
        ["Hartmann"]=500 \
        ["HolderTable"]=100 \
        ["Levy"]=90 \
        ["Powell"]=150 \
        ["SixHumpCamel"]=50 \
        ["StyblinskiTang"]=50 \
        ["SynGP"]=75 \
)

# Declare a hash map containing all the iterations for each environment
declare -A env_initial_points
env_initial_points=( \
        ["Ackley"]=20 \
        ["Alpine"]=50 \
        ["Beale"]=40 \
        ["Branin"]=10 \
        ["Cosine8"]=50 \
        ["EggHolder"]=35 \
        ["Griewank"]=8 \
        ["Hartmann"]=100 \
        ["HolderTable"]=20 \
        ["Levy"]=40 \
        ["Powell"]=35 \
        ["SixHumpCamel"]=20 \
        ["StyblinskiTang"]=30 \
        ["SynGP"]=25 \
)

# Declare a hash map containing all the lookahead steps for each algorithm in each environment
declare -A algo_lookahead
algo_lookahead=( \
        ["Ackley,HES"]=20 ["Ackley,qMSL"]=3 ["Ackley,qSR"]=0 ["Ackley,qEI"]=0 \
        ["Ackley,qPI"]=0 ["Ackley,qUCB"]=0 ["Ackley,qKG"]=0 \
        ["Alpine,HES"]=20 ["Alpine,qMSL"]=3 ["Alpine,qSR"]=0 ["Alpine,qEI"]=0 \
        ["Alpine,qPI"]=0 ["Alpine,qUCB"]=0 ["Alpine,qKG"]=0 \
        ["Beale,HES"]=20 ["Beale,qMSL"]=3 ["Beale,qSR"]=0 ["Beale,qEI"]=0 \
        ["Beale,qPI"]=0 ["Beale,qUCB"]=0 ["Beale,qKG"]=0 \
        ["Branin,HES"]=20 ["Branin,qMSL"]=3 ["Branin,qSR"]=0 ["Branin,qEI"]=0 \
        ["Branin,qPI"]=0 ["Branin,qUCB"]=0 ["Branin,qKG"]=0 \
        ["Cosine8,HES"]=20 ["Cosine8,qMSL"]=3 ["Cosine8,qSR"]=0 ["Cosine8,qEI"]=0 \
        ["Cosine8,qPI"]=0 ["Cosine8,qUCB"]=0 ["Cosine8,qKG"]=0 \
        ["EggHolder,HES"]=20 ["EggHolder,qMSL"]=3 ["EggHolder,qSR"]=0 ["EggHolder,qEI"]=0 \
        ["EggHolder,qPI"]=0 ["EggHolder,qUCB"]=0 ["EggHolder,qKG"]=0 \
        ["Griewank,HES"]=20 ["Griewank,qMSL"]=3 ["Griewank,qSR"]=0 ["Griewank,qEI"]=0 \
        ["Griewank,qPI"]=0 ["Griewank,qUCB"]=0 ["Griewank,qKG"]=0 \
        ["Hartmann,HES"]=20 ["Hartmann,qMSL"]=3 ["Hartmann,qSR"]=0 ["Hartmann,qEI"]=0 \
        ["Hartmann,qPI"]=0 ["Hartmann,qUCB"]=0 ["Hartmann,qKG"]=0 \
        ["HolderTable,HES"]=20 ["HolderTable,qMSL"]=3 ["HolderTable,qSR"]=0 ["HolderTable,qEI"]=0 \
        ["HolderTable,qPI"]=0 ["HolderTable,qUCB"]=0 ["HolderTable,qKG"]=0 \
        ["Levy,HES"]=20 ["Levy,qMSL"]=3 ["Levy,qSR"]=0 ["Levy,qEI"]=0 \
        ["Levy,qPI"]=0 ["Levy,qUCB"]=0 ["Levy,qKG"]=0 \
        ["Powell,HES"]=20 ["Powell,qMSL"]=3 ["Powell,qSR"]=0 ["Powell,qEI"]=0 \
        ["Powell,qPI"]=0 ["Powell,qUCB"]=0 ["Powell,qKG"]=0 \
        ["SixHumpCamel,HES"]=20 ["SixHumpCamel,qMSL"]=3 ["SixHumpCamel,qSR"]=0 ["SixHumpCamel,qEI"]=0 \
        ["SixHumpCamel,qPI"]=0 ["SixHumpCamel,qUCB"]=0 ["SixHumpCamel,qKG"]=0 \
        ["StyblinskiTang,HES"]=20 ["StyblinskiTang,qMSL"]=3 ["StyblinskiTang,qSR"]=0 ["StyblinskiTang,qEI"]=0 \
        ["StyblinskiTang,qPI"]=0 ["StyblinskiTang,qUCB"]=0 ["StyblinskiTang,qKG"]=0 \
        ["SynGP,HES"]=20 ["SynGP,qMSL"]=3 ["SynGP,qSR"]=0 ["SynGP,qEI"]=0 \
        ["SynGP,qPI"]=0 ["SynGP,qUCB"]=0 ["SynGP,qKG"]=0 \
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
                          --algo_n_initial_points ${env_initial_points[$env]} \
                          --algo_lookahead_steps ${algo_lookahead[$env,$algo]} \
                          --algo_ts &

        if [ "$algo" == "HES" ]
        then
            idx=$((idx+1))
            python _0_main.py --gpu_id ${gpu_id[$idx]} \
                              --algos $algo \
                              --env_names $env \
                              --seeds ${seeds[@]} \
                              --algo_n_iterations ${env_iterations[$env]} \
                              --algo_n_initial_points ${env_initial_points[$env]} \
                              --algo_lookahead_steps 1 \
                              --algo_ts &
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
#                   --algos qEI \
#                   --env_names SynGP \
#                   --seeds 2 \
#                   --algo_n_iterations 75 \
#                   --algo_lookahead_steps 0 \
#                   --algo_n_initial_points 25 \
#                   --algo_ts
#                   --test_only \
#                   --continue_once "results/exp_HES_SynGP_15" &

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
