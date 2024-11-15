# Start embedding server
python emb_server.py --model google/gemma-7b --host 0.0.0.0 --port 1337 --batch_size=8

# Sample experiment scripts
## First export machine type and mutant version
export MACHINE=ampere
export MUTANT=v1

## Myopic EI
python main.py --config configs/ei-1seq-64rs-s42.yaml

## Myopic HES-TS-AM
python main.py --config configs/hes_ts_am-1seq-64rs-s42.yaml