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
source shell/add_pwd_to_pythonpath.sh
python examples/opt_synthfunc.py --algo='hes_mc'
python examples/opt_synthfunc.py --algo='hes_vi'
python examples/opt_synthfunc.py --algo='random'
python examples/opt_synthfunc.py --algo='qEI'
python examples/opt_synthfunc.py --algo='qPI'
python examples/opt_synthfunc.py --algo='qSR'
python examples/opt_synthfunc.py --algo='qUCB'