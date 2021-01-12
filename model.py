# +
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# class edge2node:
#     def __init__(self, embedding_dim = 64):
        
# class node2edge(nn.Module):
#     def __init__(self, input_shape = (6,64), output_dim = 64, output_layer)
#         nn.Conv1d(in_channels = 6,out_channels = 1, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = False)
class compute_TE(nn.Module):
    def __init__(self, embedding_dim = 64, h_dim = 64, num_layers = 1, k = 4, dropout = 0.0, device = 'cuda'):
        super(compute_TE, self).__init__()
        
        self.embedding = embedding_dim
        self.xy_embedding = nn.Linear(2, embedding_dim)
        self.speed_embedding = nn.Linear(2, embedding_dim)
        self.final_xy_embedding = nn.Linear(2, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.densenet = models.densenet161()
        
#         self.vgg16 = models.vgg16()
        self.resnet =  nn.Sequential(models.resnet18(), nn.Linear(1000,64))
        self.resnet[0].conv1 = nn.Conv2d(in_channels = 1,out_channels = 64, kernel_size = (7,7), 
                                         stride = (2,2), padding = (3,3), bias = False)

        
        self.node2edgeConv = nn.Conv1d(in_channels = 3,out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.lastmlp = nn.Linear(embedding_dim, 1)
        self.node2edge =nn.Sequential(self.node2edgeConv, self.lastmlp)        
        
        self.loss = nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss()
    def forward(self, TE, xy, speed,load_map,is_split=True, hidden_size = 64, device = "cuda"):
        """
            xy(batch, time_length, 2)
            spped(batch, time_length, 2)
            load_map(512,512)
            is_split: input map with clipping(Only the surrounding area is observed)

        """
        max_TE = torch.max(TE)
        if max_TE > 0.001:
            TE = TE*(1/torch.max(TE))
        ## Embedding
        num_attendent = xy.shape[0]
        embedding_vec = self.speed_embedding(speed.view(-1,2)).view(-1,num_attendent, self.embedding)
        
        last_position_embedding = self.final_xy_embedding(xy[...,-1]) # n, 64
        
        ### LSTM
        state_tuple = (torch.zeros((1, num_attendent, self.embedding)).to(device), 
                       torch.zeros((1, num_attendent, self.embedding)).to(device))
        output, state = self.lstm(embedding_vec, state_tuple) # state[0] == output[-1]

        ### MAP embedding
        map_embedding = self.resnet(load_map.unsqueeze(0).unsqueeze(0)) # 1, 64
        
        
        node = output[-1] #(n, 64)
        
        predictTE = torch.zeros(len(node),len(node)).to(device)
        #compute edge
        for i in range(len(node)):
            for j in range(i+1, len(node)):
#                 print(self.node2edge(torch.cat((node[i].unsqueeze(0), node[j].unsqueeze(0), map_embedding)))) # 3, 64)
                predictTE[i][j] = self.node2edge(\
                                    torch.cat((node[i].unsqueeze(0), node[j].unsqueeze(0), map_embedding)).unsqueeze(0)\
                                                        ).squeeze().item() # 3, 64
                predictTE[j][i] = self.node2edge(\
                                    torch.cat((node[j].unsqueeze(0), node[i].unsqueeze(0), map_embedding)).unsqueeze(0)\
                                                        ).squeeze().item() # 3, 64
    
        row = (torch.argmax(predictTE, dim = 1) == torch.argmax(TE, dim = 1))
        column = (torch.argmax(predictTE, dim = 0) == torch.argmax(TE, dim = 0))
        acc = torch.sum(torch.cat((row,column))) / (len(row)*2.0)
        
#         loss = self.CEloss(torch.argmax(predictTE, dim = 1), torch.argmax(TE, dim = 1))
#         loss += self.CEloss(torch.argmax(predictTE, dim = 0), torch.argmax(TE, dim = 0))
        
        return self.loss(predictTE, TE), acc
#         return loss, acc
    
#     def node2edge():
#         return nn.Conv1d(in_channels = 6,out_channels = 1, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = False)
    
    def compute_loss():
        pass
    
    def compute_acc():
        pass
