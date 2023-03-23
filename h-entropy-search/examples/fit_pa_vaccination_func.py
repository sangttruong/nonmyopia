import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
import pickle
import pdb

region_id = 42
# 6: california, 42: pennsylvania, 53: washington, 26: new york

pos, target = pickle.load(open(f'/atlas/u/lantaoyu/projects/h-entropy/satellite_exps/data/vaccination_processed_{region_id}.pkl', 'rb'))

print(pos.shape)
print(target.shape)

# normalize the data
x_min = pos[:, 0].min()
x_max = pos[:, 0].max()
pos[:, 0] = (pos[:, 0] - x_min) / (x_max - x_min)
y_min = pos[:, 1].min()
y_max = pos[:, 1].max()
pos[:, 1] = (pos[:, 1] - y_min) / (y_max - y_min)
t_min = target.min()
t_max = target.max()
target = (target - t_min) / (t_max - t_min)
print(x_min, x_max, y_min, y_max, t_min, t_max)
#  -80.51730374552595 -74.70063627273666 39.72617096105741 42.22784028935242 0.10187117 88.13841
print("After normalization")
print(pos.min(), pos.max(), target.min(), target.max())


# Fit the func using training set
func_type = 'gp' # 'gp' or 'mlp'

if func_type == 'gp':
    func = GaussianProcessRegressor()

    sub_len = 1000
    sub_idx = np.random.choice(len(pos), size=sub_len, replace=False)
    pos = pos[sub_idx, :]
    target = target[sub_idx]

elif func_type == 'mlp':
    func = MLPRegressor(hidden_layer_sizes=(1000, 1000, 100), max_iter=1000)

print(f'Fitting func on data of sizes: pos = {pos.shape}, target = {target.shape}')
func.fit(pos, target)

# Pickle the func
filename = f'pa_vaccination_func_{func_type}.pkl'
pickle.dump(func, open(filename, 'wb'))
print(f'Fit func and saved file: {filename}')
