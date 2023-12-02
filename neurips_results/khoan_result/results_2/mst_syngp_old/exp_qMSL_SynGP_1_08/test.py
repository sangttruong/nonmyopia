import torch
import pickle

metrics = pickle.load(open("/dfs/user/sttruong/KhoanWorkspace/nonmyopia_2/results/exp_qMSL_SynGP_4_01/metrics.pkl", 'rb'))
# keys = list(metrics.keys())
# keys.sort(reverse=True, key= lambda x: int(x.split("_")[-1]))

# seed = keys[0].split("_")[-1]
# print(seed)
print(metrics)

