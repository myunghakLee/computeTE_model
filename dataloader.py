# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class dataloader(Dataset):
    def __init__(self, map_root, json_root, TE_matrix_size = 8, time_length = 10000, time_interval = 5000, MPT = 5000):  #MPT : minimal participation time
        json_root += "*/TE_AVG/*"
        map_files = glob(map_root)
        json_files = glob(json_root)
        self.data = []
        for json_file in json_files:
            with open(map_root + json_file.split('/')[-3] + ".json") as json_data:
                map_array = np.array(json.load(json_data))
            with open(json_file) as json_data:
                data = json.load(json_data)
            for scene in data['scene_attendent'].keys():  # json 파일안의 scene 하나
                ids = np.array(data['scene_attendent'][scene]['id']) - 1
                TE = np.array(data['TEmatrix'][scene])
                start_time, end_time= int(scene.split("_")[0]), int(scene.split("_")[-1])
                
                for time in range(start_time, end_time, time_interval): # 각 scene을 time_interval간격으로 나눔
                    attendent= []
                    attend_TE = []
                    for id in ids:
                        if self.attendent_check(time, time+time_length, data['data'][id]['time']):
                            attendent.append(id)
                    
                    if len(attendent) > 2:
                        initial_data = {}
                        initial_data["TE"] = np.array([TE[np.where(ids == a)[0][0]] for a in attendent])
                        initial_data["TE"] =  np.array([initial_data["TE"][:,np.where(ids == a)[0][0]] for a in attendent])
                        initial_data["xy"] = []
                        for a in attendent:
                            xy = data['data'][a]['xy']
                            prefix = []
                            postfix = []
                            split_start = 0
                            split_end = -1
                            if data['data'][a]['time'][-1] < time + time_length:
                                plus_num = int((time + time_length - data['data'][a]['time'][-1])/100)
                                postfix = [[0,0]]* plus_num
                            else:
                                try:
                                    split_end = data['data'][a]['time'].index(time + time_length)
                                except:
                                    print(time, time_length)
                                    print(data['data'][a]['time'])
                                    assert False, "AA"
                                xy = xy[:split_end + 1]

                            if data['data'][a]['time'][0] > time:
                                plus_num = int((data['data'][a]['time'][0] - time)/100)
                                prefix = [[0,0]]* plus_num
                            else:
                                split_start = data['data'][a]['time'].index(time)
                                xy = xy[split_start:]

                            initial_data["xy"].append(prefix + xy + postfix)
                            
                        initial_data["xy"] =np.transpose(initial_data["xy"], (0,2,1))
                        initial_data["xy"] = np.array(initial_data["xy"])
                        initial_data["map"] = map_array
                        self.data.append(initial_data)
                
                
                

        
    def attendent_check(self,start, end, arr, threshold = 4000):
        if len(arr) < threshold / 100:
            return False
        if (start < arr[0] < end) and (arr[0] + threshold < end):
            return True
        elif (start < arr[-1] < end) and (start + threshold < arr[-1]):
            return True
        return False
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# +
# json_root = "dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/"
# map_root = "dataset/INTERACTION-Dataset-DR-v1_1/maps/map_arr/"
# A = dataloader(map_root, json_root)


# +
# len(A)
# -

from matplotlib import pyplot as plt
def saniticheck(data):
    agent = 2
    mul = 1/max(data["TE"][agent])
    
    for i, xy_data in enumerate(data['xy']):
        while(xy_data[0][0] == 0):
            xy_data = xy_data[1:]
        while(xy_data[-1][0] == 0):
            xy_data = xy_data[:-1]
        if i ==agent:
            plt.scatter(xy_data[:,0][-1],xy_data[:,1][-1], color = 'black')
            plt.plot(xy_data[:,0],xy_data[:,1], color = 'black', zorder = -1)
        else:
            plt.scatter(xy_data[:,0][-1],xy_data[:,1][-1])
            plt.plot(xy_data[:,0],xy_data[:,1])
            
        plt.scatter(xy_data[:,0][-1],xy_data[:,1][-1], color = 'red', s= 60, alpha = data['TE'][agent,i]*mul)


# +

# saniticheck(A[9])
