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


# -

train_idx = int(len(data) // 10 * 9)
val_idx = int(len(data) // 10 * 9.5)
test_idx = len(data)


indices = torch.randperm(len(data)).tolist()
dataset_train = torch.utils.data.Subset(data, indices[:train_idx])
dataset_val = torch.utils.data.Subset(data, indices[train_idx:val_idx])
dataset_test = torch.utils.data.Subset(data, indices[val_idx:])

# +
dataset_loader_train = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=1, shuffle=True,
                                              num_workers=0)
dataset_loader_val = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=1, shuffle=False,
                                              num_workers=0)
dataset_loader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1, shuffle=False,
                                              num_workers=0)


TEmodel = model.compute_TE()
TEmodel.to(device)
optimizer = optim.Adam(TEmodel.parameters(), lr=0.0001)
criterion = nn.MSELoss()


# +
acc_all = []
loss_all = []
low_acc = []
high_acc = []

decay = 1.0
for epoch in range(500):
    acc_arr = []
    val_acc = []
    if epoch % 4 == 3:
        decay *= 0.9
    for i, batch in tqdm(enumerate(dataset_loader_train)):
        

        xy,load_map,TE = batch
        
        for i in range(len(TE)):
            for j in range(len(TE)):
                if TE[0][i][j] < 0:
                    TE[0][i][j] *= 0.0
        
        
        max_TE = torch.max(TE)
        if max_TE > 0.01:
            TE /= max_TE
        else:
            TE -= TE
            
        
        xy = xy[0].to(device)
        TE = TE[0].to(device)
        load_map = load_map[0].to(device)
        
        speed = torch.cat((torch.zeros((xy.shape[0],2,1)).to(device),
                               xy[:,:,1:] - xy[:,:,:-1]), dim = 2).to(device)
        
        
        optimizer.zero_grad()
        predictTE, N = TEmodel(xy,speed, load_map)
#         predictTE = predictTE.reshape(len(TE),len(TE)).T
        for i in range(len(predictTE)):
            predictTE[i][i] -= predictTE[i][i]

        predictTE_row = torch.argsort(predictTE, dim = 1).to(device).float()
        predictTE_col = torch.argsort(predictTE, dim = 0).to(device).float()

        TE_row = torch.argsort(TE, dim = 1).to(device).float()
        TE_col = torch.argsort(TE, dim = 0).to(device).float()

#         loss = criterion(predictTE - predictTE + TE_row, TE - TE + TE_row) / 0.5
#         loss += criterion(predictTE - predictTE + predictTE_col,TE - TE+ TE_col) / 0.5
#         loss *=decay
        row = (torch.argmax(predictTE, dim = 1) == torch.argmax(TE, dim = 1))
        column = (torch.argmax(predictTE, dim = 0) == torch.argmax(TE, dim = 0))
        acc = torch.sum(torch.cat((row,column))) / (len(row)*2.0)

        acc_arr.append(acc.item())

#         print(decay)
#         print(TE)
#         print(predictTE)
#         print(criterion(predictTE, TE))
#         print(criterion(predictTE, TE) * decay)
#         assert False, "SS"
        
        
        loss = criterion(predictTE, TE) * decay
        loss_all.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} loss : ", np.mean(loss_all))
    print(f"epoch {epoch} Accuracy : ", np.mean(acc_arr))

    acc_arr = []
    
    for i, batch in tqdm(enumerate(dataset_loader_val)):
        with torch.no_grad():
            xy,load_map,TE = batch
            xy = xy[0].to(device)
            TE = TE[0].to(device)
            load_map = load_map[0].to(device)
            max_TE = torch.max(TE)
            if max_TE > 0.0001:
                TE /= max_TE
            else:
                TE -= TE
            speed = torch.cat((torch.zeros((xy.shape[0],2,1)).to(device),
                                   xy[:,:,1:] - xy[:,:,:-1]), dim = 2).to(device)

            predictTE, N = TEmodel(xy,speed, load_map)
            for i in range(len(predictTE)):
                predictTE[i][i] -= predictTE[i][i]
            predictTE = predictTE.reshape(len(TE),len(TE))

            
            row = (torch.argmax(predictTE, dim = 1) == torch.argmax(TE, dim = 1))
            column = (torch.argmax(predictTE, dim = 0) == torch.argmax(TE, dim = 0))
            acc = torch.sum(torch.cat((row,column))) / (len(row)*2.0)
            if acc < 0.2:
                low_acc.append([predictTE, TE])
            elif acc > 0.6:
                high_acc.append([predictTE, TE])
                
            acc_arr.append(acc.item())
            acc_all.append(acc.item())
    print(f"epoch {epoch} Accuracy : ", np.mean(acc_arr))
    val_acc.append(np.mean(acc_arr))
#     print(TE, predictTE)
# -


predictTE[0].unsqueeze(0)

low_acc[-5]

high_acc[-18]

# +
from matplotlib import pyplot as plt
idx = -5
df = low_acc[idx][0].tolist()
plt.title("predictTE")
plt.pcolor(df)
plt.colorbar()
plt.show()

df = low_acc[idx][1].tolist()
plt.title("groundTruth")
plt.pcolor(df)
plt.colorbar()
plt.show()



# +
from matplotlib import pyplot as plt
idx = 30
df = high_acc[idx][0].tolist()
plt.title("predictTE")
plt.pcolor(df)
plt.colorbar()
plt.show()

df = high_acc[idx][1].tolist()
plt.title("groundTruth")
plt.pcolor(df)
plt.colorbar()
plt.show()


# -

high_acc[5]

loss

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
# plt.plot(np.array(acc_all_avg[:]), label = 'acc')
plt.plot(np.array(loss_all_avg[80:]), label = 'loss')
plt.legend()
plt.show()

from matplotlib import pyplot as plt
# plt.plot(np.array(acc_all_avg[:]), label = 'acc')
plt.plot(np.array(val_acc), label = 'loss')
plt.legend()
plt.show()



# +
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def init_adj(num_ped):
    # rel_rec: [N_edges, num_ped]
    # rel_send: [N_edges, num_ped]
    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_ped, num_ped])

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()

    return rel_rec, rel_send


def edge2node(self, x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    incoming = torch.matmul(rel_rec.t(), x)
    return incoming / incoming.size(1)

def node2edge(self, x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    receivers = torch.matmul(rel_rec, x)
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([receivers, senders], dim=1)
    return edges



# -

criterion(predictTE, TE) * decay

rel_rec, rel_send = init_adj(4)


rel_rec, rel_send = self.init_adj(num_ped)


SM = nn.Softmax()

TE

SM(TE[0])

torch.max(TE)

import numpy as np
from scipy.stats import zprob
def zTransform(r, n):
    z = np.log((1 + r) / (1 - r)) * (np.sqrt(n - 3) / 2)
    p = zprob(-z)
    return p


loss = nn.MarginRankingLoss()
input1 = torch.randn(3, requires_grad=True)
input2 = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()
output = loss(input1, input2, target)
output.backward()

# +
input1 = torch.tensor([3,2,1])
input2 = torch.tensor([11,22,33])

output = loss(input1, input2, target)

# -

torch.eye(10)

# +

torch.matrix_rank(torch.eye(3))

