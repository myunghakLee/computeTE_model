# -*- coding: utf-8 -*-
# +
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import resnet

class embedding_map_net(nn.Module):
    def __init__(self):
        super(embedding_map_net, self).__init__()
        

        self.ResidualBlock1 = resnet.ResidualBlock(1,4, stride = 2, downsample = resnet.downsample_layer(1,4,2))
        self.ResidualBlock2 = resnet.ResidualBlock(4,8, stride = 2, downsample = resnet.downsample_layer(4,8,2))
        self.ResidualBlock3 = resnet.ResidualBlock(8,16, stride = 2, downsample = resnet.downsample_layer(8,16,2))
        self.ResidualBlock4 = resnet.ResidualBlock(16,32, stride = 2, downsample = resnet.downsample_layer(16,32,2))

        self.conv_1 = nn.Conv2d(32,64,3, stride = 2)

        self.average_pooling = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.ResidualBlock1(x)
        x = self.ResidualBlock2(x)
        x = self.ResidualBlock3(x)
        x = self.ResidualBlock4(x)
        x = self.conv_1(x)
        x = self.average_pooling(x)
        return x


# +
def repeat(tensor, num_reps):
    """
    Inputs:
    -tensor: 2D tensor of any shape
    -num_reps: Number of times to repeat each row
    Outpus:
    -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    """
    col_len = tensor.size(1)
    tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
    tensor = tensor.view(-1, col_len)
    return tensor
a = torch.tensor([[1,20,3],[4,5,6]])
# a = a.repeat(2,1)


q1 = a.repeat(2,1)
q2 = repeat(a, 2)
q1-q2
a = a.repeat(2,1)
a
# +

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

        
        self.node2edgeConv = nn.Conv1d(in_channels = 4,out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.lastmlp = nn.Linear(embedding_dim, 1)
        self.node2edge =nn.Sequential(self.node2edgeConv,
                                      nn.ReLU(), 
                                      self.lastmlp)        
        
        self.loss = nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss()
        self.AA = nn.Linear(64, 11)
        self.map_embedding_layer = embedding_map_net()
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        
        self.last_mlp1 = nn.Linear(128,64)
        self.last_mlp2 = nn.Linear(64,16)
        self.last_mlp3 = nn.Linear(16,4)
        self.last_mlp4 = nn.Linear(4,1)
#         self.last_mlp5 = nn.Linear(8,1)
        self.last_embedding = nn.Sequential(
                            self.last_mlp1,
                            nn.ReLU(),
                            self.last_mlp2,
                            nn.ReLU(),
                            self.last_mlp3,
                            nn.ReLU(),
                            self.last_mlp4
        )
        
    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor
        
    def forward(self, xy, speed,load_map,is_split=True, hidden_size = 64, device = "cuda"):
        """
            xy(batch, time_length, 2)
            spped(batch, time_length, 2)
            load_map(512,512)
            is_split: input map with clipping(Only the surrounding area is observed)

        """
        ## Embedding
        num_attendent = xy.shape[0]
        embedding_vec = self.speed_embedding(speed.view(-1,2)).view(-1,num_attendent, self.embedding)
        last_pos = xy[...,-1] # n, 64
        
        ### LSTM
        state_tuple = (torch.zeros((1, num_attendent, self.embedding)).to(device), 
                       torch.zeros((1, num_attendent, self.embedding)).to(device))
        output, state = self.lstm(embedding_vec, state_tuple) # state[0] == output[-1]

        ### MAP embedding
#         map_embedding = self.resnet(load_map.unsqueeze(0).unsqueeze(0)) # 1, 64
        map_embedding = self.map_embedding_layer(load_map.reshape(1,1,512,512))
        
        node = output[-1] #(n, 64)
        attendent_num = len(node)
        node = node.repeat(attendent_num,1) # (1,2,3) --> (1,2,3,1,2,3,1,2,3)
        
        last_pos1 = last_pos.repeat(attendent_num,1) # (1,2,3) --> (1,2,3,1,2,3,1,2,3)
        last_pos2 = self.repeat(last_pos, attendent_num) # (1,2,3) --> (1,1,1,2,2,2,3,3,3)
        curr_rel_pos = last_pos1 - last_pos2
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
        
        
        # a와a의 상대적 위치 그리고 a의 embedding
        # a와b의 상대적 위치 그리고 b의 embedding
        # a와c의 상대적 위치 그리고 c의 embedding
        # ...
        mlp_input = torch.cat([curr_rel_embedding, node], dim=1) 
        
        predictTE = self.last_embedding(mlp_input)

        return predictTE, node

    def calc_edge(self, node1, node2,final_rel):
        return self.node2edge(torch.cat((node1.unsqueeze(0), node2.unsqueeze(0), final_rel.unsqueeze(0))).unsqueeze(0)).squeeze()
        
        
    def compute_loss():
        pass
    
    def compute_acc():
        pass
# -


