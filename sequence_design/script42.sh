export LIBRARY_PATH=/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib/python3.10/site-packages/torch/lib:/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib/python3.10/site-packages/torch/lib:/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2
python main.py --config ready_configs/hes_ts_am-1seq-128rs-s42.yaml
python main.py --config ready_configs/sr-1seq-128rs-s42.yaml