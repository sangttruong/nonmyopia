#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --job-name="hes"
#SBATCH --output=sbatch_output/%j.out

source ~/.bashrc
conda activate botorch
cd /atlas/u/lantaoyu/projects/h-entropy
source shell/add_pwd_to_pythonpath.sh
python examples/multilevelset_alpine.py --seed=10 --algo=rs
python examples/multilevelset_alpine.py --seed=20 --algo=rs
python examples/multilevelset_alpine.py --seed=30 --algo=rs
python examples/multilevelset_alpine.py --seed=10 --algo=us
python examples/multilevelset_alpine.py --seed=20 --algo=us
python examples/multilevelset_alpine.py --seed=30 --algo=us
python examples/multilevelset_alpine.py --seed=10 --algo=kg
python examples/multilevelset_alpine.py --seed=20 --algo=kg
python examples/multilevelset_alpine.py --seed=30 --algo=kg
python examples/multilevelset_alpine.py --seed=10 --algo=hes
python examples/multilevelset_alpine.py --seed=20 --algo=hes
python examples/multilevelset_alpine.py --seed=30 --algo=hes
