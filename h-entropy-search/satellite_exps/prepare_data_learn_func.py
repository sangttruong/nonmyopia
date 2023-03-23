import os
import sys
import torch
import numpy as np
from matplotlib import cm, pyplot as plt

from utils.utils import *
from utils.constants import CUTSIZEX, CUTSIZEY, AREA, GT_MS_COUNT, US_STATES, AFRICAN_COUNTRIES
import pdb
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from src.utilities import MLP

#######################################################################
# country = "africa"  # "us" or "africa"
# district = "nigeria"  # check utils.constants.py
country = "us"  # "us" or "africa"
district = "pennsylvania"  # pennsylvania
sampling_method = "NL"  # nightlight
# sampling_method = "population"  # population
#######################################################################

# Directories to the covariate data
nl_data = "/atlas/u/chenlin/objectcount/images/dnb_land_ocean_ice.2012.54000x27000_geo.tif"
pop_data = '/atlas/u/jesslec/ObjectCount/object_count/notebooks/visualization/population/gpw_v4_population_density_rev11_2020_30_sec.tif'

print("loading data")
if sampling_method == "NL":
    raster_nl = rs.open(nl_data)
    raster_nl_img = load_geotiff(nl_data)
    r_data = raster_nl_img
else:
    raster_pop = rs.open(pop_data)
    raster_pop_img = load_geotiff(pop_data)
    r_data = raster_pop_img
print("Data loaded")

print(f"processing {country} {district} {sampling_method}", flush=True)
if country == "africa":
    if sampling_method == "NL":
        cutsizex = CUTSIZEX[sampling_method][district]
        cutsizey = CUTSIZEY[sampling_method][district]
    else:
        cutsizex = CUTSIZEX[sampling_method]["africa"]
        cutsizey = CUTSIZEY[sampling_method]["africa"]
else:
    cutsizex = CUTSIZEX[sampling_method]["us"]
    cutsizey = CUTSIZEY[sampling_method]["us"]

if district == "alaska":
    if sampling_method == "NL":
        cutsizex = [2000, 6000]
        cutsizey = [0, 8000]
    elif sampling_method == "population":
        cutsizex = [2000, 6000]
        cutsizey = [0, 6500]
if district == "hawaii":
    if sampling_method == "NL":
        cutsizex = [10000, 11000]
        cutsizey = [2500, 4000]
    elif sampling_method == "population":
        cutsizex = [8000, 9000]
        cutsizey = [2000, 3500]

print("Country {}, district {}".format(country, district))
log_root = "/atlas/u/jesslec/ObjectCount/object_count/notebooks/visualization"
pth_mask = f'{log_root}/saved_data/{sampling_method}/{cutsizex[0]}_{cutsizex[1]}_{cutsizey[0]}_{cutsizey[1]}_{district}_mask.pth'
binary_m = torch.load(pth_mask)

if not os.path.isfile(pth_mask):
    print("mask {} not exist {} {}".format(pth_mask, country, district), flush=True)
else:
    print("mask loaded")

img = r_data[cutsizex[0]:cutsizex[1], cutsizey[0]: cutsizey[1]]  # only unmasked pixels
# if sampling_method == "NL":
#     plt.imshow(img)
# else:
#     plt.imshow(np.log(np.clip(img, a_min=1e-20, a_max=1e20)))
# plt.savefig(f"./figures/{country}_{district}_{sampling_method}.png")

# get bounding box
start_x, end_x, start_y, end_y = None, None, None, None
for i in range(binary_m.shape[1]):
    if start_x is None:
        if True in binary_m[:, i]:
            start_x = i
            break
for i in reversed(range(binary_m.shape[1])):
    if end_x is None:
        if True in binary_m[:, i]:
            end_x = i
            break
for i in range(binary_m.shape[0]):
    if start_y is None:
        if True in binary_m[i, :]:
            start_y = i
            break
for i in reversed(range(binary_m.shape[0])):
    if end_y is None:
        if True in binary_m[i, :]:
            end_y = i
            break

img = img * binary_m
img = img[start_y:end_y, start_x:end_x]
if sampling_method == "population":
    img = np.log(np.clip(img, a_min=1e-10, a_max=1e10))
plt.imshow(img)
plt.savefig(f"./figures/{country}_{district}_{sampling_method}.png")
np.save(open(f"./figures/{country}_{district}_{sampling_method}.npy", "wb"), img)
exit(0)

# Make Training Set
pos = []
target = []
y_lim = img.shape[0] - 1
x_lim = img.shape[1] - 1
target_max = float(img.max())
target_min = float(img.min())

for y in range(y_lim):
    for x in range(x_lim):
        pos.append([float(x) / x_lim, 1 - float(y) / y_lim])  # left bottom corner is the origin
        target.append((img[y, x] - target_min) / (target_max - target_min))
pos = torch.tensor(pos)
target = torch.tensor(target)

# Start training MLP
batch_size = 1000
epoch_num = 20
learning_rate = 1e-2
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
        loss = ((value - outputs) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
    print(f"Epoch: {epoch} Avg_loss: {np.mean(loss_list)}")

model_path = f"./models/{country}_{district}_{sampling_method}.pt"
torch.save(model.state_dict(), model_path)
