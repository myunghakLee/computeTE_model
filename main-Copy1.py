import model
import torch
import dataloader
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn


# +
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

json_root = "dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/"
map_root = "dataset/INTERACTION-Dataset-DR-v1_1/maps/map_arr/"

data = dataloader.dataloader(map_root, json_root)
dataset_loader = torch.utils.data.DataLoader(data,
                                              batch_size=1, shuffle=False,
                                              num_workers=0)
TEmodel = model.compute_TE()
TEmodel.to(device)
optimizer = optim.Adam(TEmodel.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# +
acc_all = []
loss_all = []
for epoch in range(500):
    acc_arr = []
    for i, batch in tqdm(enumerate(dataset_loader)):
#         xy_shape = batch["xy"].shape
#         batch["xy"] = batch["xy"].view(-1)
#         for j in range(len(batch["xy"])):
#             if batch["xy"][j] < 0:
#                 batch["xy"][j] *= 0.0
#         batch["xy"] = batch["xy"].view(xy_shape)
        xy,load_map,TE = batch
        xy = xy[0].to(device)
        TE = TE[0].to(device)
        TE += torch.min(TE)
        max_TE = torch.max(TE)
        if max_TE > 0.001:
            TE /= max_TE
        else:
            TE -=TE
        load_map = load_map[0].to(device)
        
#         xy.requires_grad=True
#         TE.requires_grad=True
#         load_map.requires_grad=True
        speed = torch.cat((torch.zeros((xy.shape[0],2,1)).to(device),
                               xy[:,:,1:] - xy[:,:,:-1]), dim = 2).to(device)
        
        
        optimizer.zero_grad()
        predictTE, N = TEmodel(xy,speed, load_map)
#         TE = torch.ones(predictTE.shape).to(device)

        loss = criterion(predictTE.squeeze(), TE)*9.0
        loss.backward()
        optimizer.step()
        
        row = (torch.argmax(predictTE, dim = 1) == torch.argmax(TE, dim = 1))
        column = (torch.argmax(predictTE, dim = 0) == torch.argmax(TE, dim = 0))
        acc = torch.sum(torch.cat((row,column))) / (len(row)*2.0)
        
        acc_arr.append(acc.item())
        acc_all.append(acc.item())
        loss_all.append(loss.item())
        if i % 900 ==899:
            
            print(np.mean(acc_arr), loss)
            acc_arr = []
#     print(TE, predictTE)
# -

torch.save(model.state_dict(), "NotU")


# +
def A(a,b):
    return torch.tensor(10*a + b)
results = A(0*10, 0).unsqueeze(0)
for i in range(1,4):
    results = torch.cat((results, A(0*10, i).unsqueeze(0)))
results = results.unsqueeze(0)


for i in range(1,3):
    output = A(i * 10, 0).unsqueeze(0)
    for j in range(1, 4):
        output = torch.cat((output, A(i*10, j).unsqueeze(0)))
    results = torch.cat((results, output.unsqueeze(0)))
# -

acc_all[:3]

acc_all

from matplotlib import pyplot as plt
plt.plot(np.array(acc_all[:]), label = 'acc')
plt.plot(np.array(loss_all[:]), label = 'loss')
plt.legend()
plt.show()

torch.max(TE)

acc_all_avg = []
loss_all_avg = []
for i in range(0,len(acc_all)-50, 50):
    acc_all_avg.append(np.mean(acc_all[i:i+50]))
    loss_all_avg.append(np.mean(loss_all[i:i+50]))

from matplotlib import pyplot as plt
plt.plot(np.array(acc_all_avg[:]), label = 'acc')
plt.plot(np.array(loss_all_avg[:]), label = 'loss')
plt.legend()
plt.savefig("acc_loss.png")
plt.show()


attendent_num = []
for d in data:
    attendent_num.append(len(d["TE"]))
np.mean(attendent_num)

loss_all[i:i+50]

loss_all_avg[:6]

node = predictTE
node.shape

torch.cat([torch.cat([[TE[i] for i in range(len(TE))]]) for j in range(2)])

[a[0],a[1],a[2]]

TE_MAT = torch.tensor(TE_MAT)

[i for i in TE_VEC]

torch.cat([[1,2,3]], dim = 2)

A = torch.tensor([A(0*10, j) for j in range(4)])
torch.cat(A)
