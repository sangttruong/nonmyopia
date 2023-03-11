import pandas as pd
from shapely.geometry import Polygon
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from src.utilities import MLP
import pickle

load_data = True
region_id = 36

if load_data:
    result = pd.read_csv('/atlas/u/shengjia/covid/data/vac_inferred_lvm_v4.csv')
    feat = pd.read_csv('/atlas/u/shengjia/covid/data/census_cbg_with_predicted_hesitancy_vaccincation.csv')

    result = result.merge(feat, on='census_block_group', how='left')
    features = result[['census_block_group', 'longitude', 'latitude', 'vac_rate_inferred']]
    print(features.head())
    pennsylvania = features[features['census_block_group'] // 10000000000 == region_id]

    print(pennsylvania.head())

    reader = json.load(open('/atlas/u/shengjia/covid/data/cbg.geojson'))
    dicts = []

    for i in range(len(reader['features'])):
        item = {
            'census_block_group': int(reader['features'][i]['properties']['CensusBlockGroup']),
            'poly': Polygon(reader['features'][i]['geometry']['coordinates'][0][0])
        }
        dicts.append(item)

    pennsylvania = pennsylvania.merge(pd.DataFrame.from_dict(dicts), on='census_block_group')
    gdf = gpd.GeoDataFrame(pennsylvania, geometry='poly')
    gdf.plot(column='vac_rate_inferred', legend=True, figsize=(10, 10),
             vmin=10, vmax=100, cmap='coolwarm_r')
    plt.savefig(f'./figures/vac_rate_{region_id}.png')
    pickle.dump(pennsylvania, open(f'./data/vaccination_{region_id}.pkl', 'wb'))
else:
    pennsylvania = pickle.load(open(f'./data/vaccination_{region_id}.pkl', 'rb'))

pos = []
target = []
for i in range(len(pennsylvania)):
    pos.append([pennsylvania['longitude'][i], pennsylvania['latitude'][i]])
    target.append(pennsylvania['vac_rate_inferred'][i])
pos = np.array(pos)
target = np.array(target)

pickle.dump([pos, target], open(f'./data/vaccination_processed_{region_id}.pkl', 'wb'))

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

pos = torch.tensor(pos).float()
target = torch.tensor(target).float()

# Start training MLP
batch_size = 100
epoch_num = 30
learning_rate = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(TensorDataset(pos, target), batch_size=batch_size, shuffle=True, drop_last=False)
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Start training MLP")
for epoch in range(epoch_num):
    loss_list = []
    for position, value in train_loader:
        position, value = position.to(device), value.to(device)
        outputs = model(position)
        loss = ((value - outputs.squeeze()) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
    print(f"Epoch: {epoch} Avg_loss: {np.mean(loss_list)}")

model_path = f"./models/vaccination.pt"
torch.save(model.state_dict(), model_path)
