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

        
        self.node2edgeConv = nn.Conv1d(in_channels = 3,out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.lastmlp = nn.Linear(embedding_dim, 1)
        self.node2edge =nn.Sequential(self.node2edgeConv,
                                      nn.ReLU(), 
                                      self.lastmlp)        
        
        self.loss = nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss()
        self.AA = nn.Linear(64, 11)
        self.map_embedding_layer = embedding_map_net()
        
        
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
        last_position_embedding = self.final_xy_embedding(xy[...,-1]) # n, 64
        
        ### LSTM
        state_tuple = (torch.zeros((1, num_attendent, self.embedding)).to(device), 
                       torch.zeros((1, num_attendent, self.embedding)).to(device))
        output, state = self.lstm(embedding_vec, state_tuple) # state[0] == output[-1]

        ### MAP embedding
#         map_embedding = self.resnet(load_map.unsqueeze(0).unsqueeze(0)) # 1, 64
        map_embedding = self.map_embedding_layer(load_map.reshape(1,1,512,512))
        
        node = output[-1] #(n, 64)
        
        
#         predictTE = self.AA(node)

#         for i in range(len(node)):
#             node[i] = torch.cat((node[i].unsqueeze(0), last_position_embedding[i].unsqueeze(0)), dim = 0)

        predictTE = self.calc_edge(node[0], node[0],last_position_embedding[0] - last_position_embedding[0]).unsqueeze(0)
        for i in range(1, len(node)):
            predictTE = torch.cat((predictTE, self.calc_edge(node[0], node[i],last_position_embedding[0] - last_position_embedding[i]).unsqueeze(0)))
        predictTE = predictTE.unsqueeze(0)

        for i in range(1,len(node)):
            output = self.calc_edge(node[i], node[0],last_position_embedding[i] - last_position_embedding[0]).unsqueeze(0)
            for j in range(1, len(node)):
                output = torch.cat((output, self.calc_edge(node[i], node[j], last_position_embedding[i] - last_position_embedding[j]).unsqueeze(0)))
            predictTE =  torch.cat((predictTE, output.unsqueeze(0)))

# #         predictTE = torch.cat([])
        
    
# #         predictTE = torch.cat([torch.cat([self.calc_edge(node[i], node[j]) for j in range(len(node))]) for i in range(len(node))])
    
# # #         predictTE.requires_grad = True
# #         #compute edge

#         TE_MAT = []
#         for i in range(1,len(node)):
#             TE_VEC = []
#             for j in range(len(node)):
#                 TE_VEC.append(self.calc_edge(node[i], node[j]))
#             TE_MAT.append(TE_VEC)
#         TE_MAT = torch.ten
# #                 predictTE[i][j] = self.node2edge(\
# #                                     torch.cat((node[i].unsqueeze(0), node[j].unsqueeze(0))).unsqueeze(0)\
# #                                                         ).squeeze() # 3, 64
# #                 predictTE[j][i] = self.node2edge(\
# #             torch.cat((node[j].unsqueeze(0), node[i].unsqueeze(0))).unsqueeze(0)\
# #                                                         ).squeeze() # 3, 64
#             TE_VEC = torch.tensor([TE_VEC])
#             predictTE = torch.cat(predictTE, TE_VEC, dim = 1)
# #         loss = self.CEloss(torch.argmax(predictTE, dim = 1), torch.argmax(TE, dim = 1))
# #         loss += self.CEloss(torch.argmax(predictTE, dim = 0), torch.argmax(TE, dim = 0))
        
        return predictTE, node

    def calc_edge(self, node1, node2,final_rel):
        return self.node2edge(torch.cat((node1.unsqueeze(0), node2.unsqueeze(0), final_rel.unsqueeze(0))).unsqueeze(0)).squeeze()
        
        
    def compute_loss():
        pass
    
    def compute_acc():
        pass
# -


