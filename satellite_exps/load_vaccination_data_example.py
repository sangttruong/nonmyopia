import numpy as np
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt

region_id = 42
# 6: california, 42: pennsylvania, 53: washington, 26: new york


# Load and process the data
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


# Plot the data
pennsylvania = pickle.load(open(f'/atlas/u/lantaoyu/projects/h-entropy/satellite_exps/data/vaccination_{region_id}.pkl', 'rb'))
gdf = gpd.GeoDataFrame(pennsylvania, geometry='poly')
gdf.plot(column='vac_rate_inferred', legend=False, vmin=0, vmax=100, cmap='coolwarm_r')  # Set legend=True if need the color bar
plt.savefig(f'./figures/vac_rate_{region_id}.png')
