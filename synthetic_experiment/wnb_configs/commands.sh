# Start new WandB sweep
wandb sweep wnb_configs/full.yaml
# Then we will get the command to start agent
# For example: wandb agent ura/nonmyopia/urcfmx4d


# Authenicate for each terminal
/afs/cs/software/bin/reauth


# Start experiments
sang
init_conda
conda activate bo
cd nonmyopia
CUDA_VISIBLE_DEVICES=0 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=2 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=3 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=4 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=5 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=8 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=9 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=1 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=6 wandb agent ura/nonmyopia/r05545gl &
CUDA_VISIBLE_DEVICES=7 wandb agent ura/nonmyopia/r05545gl &
