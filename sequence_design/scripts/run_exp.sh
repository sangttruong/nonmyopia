# Start embedding server
# 2 x 48GB
python emb_server.py --model google/gemma-7b --host 0.0.0.0 --port 1337 --batch_size=8

# Experiment scripts
python main.py --config configs/sr-1seq-64rs-s42.yaml
python main.py --config configs/sr-1seq-64rs-s45.yaml
python main.py --config configs/sr-1seq-64rs-s49.yaml

python main.py --config configs/ei-1seq-64rs-s42.yaml
python main.py --config configs/ei-1seq-64rs-s45.yaml
python main.py --config configs/ei-1seq-64rs-s49.yaml

python main.py --config configs/pi-1seq-64rs-s42.yaml
python main.py --config configs/pi-1seq-64rs-s45.yaml
python main.py --config configs/pi-1seq-64rs-s49.yaml

python main.py --config configs/ucb-1seq-64rs-s42.yaml
python main.py --config configs/ucb-1seq-64rs-s45.yaml
python main.py --config configs/ucb-1seq-64rs-s49.yaml

python main.py --config configs/kg-1seq-64rs-s42.yaml
python main.py --config configs/kg-1seq-64rs-s45.yaml
python main.py --config configs/kg-1seq-64rs-s49.yaml

python main.py --config configs/hes_ts_am-1seq-64rs-s42.yaml
python main.py --config configs/hes_ts_am-1seq-64rs-s45.yaml
python main.py --config configs/hes_ts_am-1seq-64rs-s49.yaml