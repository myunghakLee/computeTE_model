import model
import torch
import dataloader
import numpy as np
from tqdm import tqdm
import torch.optim as optim


# +
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

json_root = "dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/"
map_root = "dataset/INTERACTION-Dataset-DR-v1_1/maps/map_arr/"

data = dataloader.dataloader(map_root, json_root)
dataset_loader = torch.utils.data.DataLoader(data,
                                              batch_size=1, shuffle=True,
                                              num_workers=0)
model = model.compute_TE()
model.to(device)

# -

optimizer = optim.Adam(model.parameters(), lr=0.001)

acc_all = []
loss_all = []
for epoch in range(100):
    acc_arr = []
    for i, batch in tqdm(enumerate(dataset_loader)):
        xy_shape = batch["xy"].shape
        batch["xy"] = batch["xy"].view(-1)
        for j in range(len(batch["xy"])):
            if batch["xy"][j] < 0:
                batch["xy"][j] *= 0.0
        batch["xy"] = batch["xy"].view(xy_shape)

        xy = batch["xy"][0].float().to(device)
        TE = batch["TE"][0].float().to(device)
        load_map = batch["map"][0].float().to(device)
        xy.requires_grad=True
        TE.requires_grad=True
        load_map.requires_grad=True
        speed = torch.cat((torch.zeros((xy.shape[0],2,1)).to(device),
                               xy[:,:,1:] - xy[:,:,:-1]), dim = 2).to(device)
        optimizer.zero_grad()
        loss, acc = model(TE, xy,speed, load_map)
        loss.backward()
        optimizer.step()
        acc_arr.append(acc.item())
        acc_all.append(acc.item())
        loss_all.append(loss.item())
        if i % 300 ==0:
            
            print(np.mean(acc_arr), loss)
            acc_arr = []


attendent_num = []
for d in data:
    attendent_num.append(len(d["TE"]))
np.mean(attendent_num)

acc_all_avg = []
loss_all_avg = []
for i in range(0,len(acc_all)-50, 50):
    acc_all_avg.append(np.mean(acc_all[i:i+50]))
    loss_all_avg.append(np.mean(loss_all[i:i+50]))

from matplotlib import pyplot as plt
plt.plot(np.array(acc_all_avg[:]), label = 'acc')
plt.plot(np.array(loss_all_avg[:]), label = 'loss')
plt.legend()
plt.show()

from matplotlib import pyplot as plt
plt.plot(np.array(acc_all_avg[:]), label = 'acc')
plt.plot(np.array(loss_all_avg[:]), label = 'loss')
plt.legend()
plt.show()
