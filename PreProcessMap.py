# -*- coding: utf-8 -*-
# +
from xml.etree.ElementTree import parse
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import json
import os

file_num = 1
file_root = "dataset/INTERACTION-Dataset-DR-v1_1/maps/"
file_list = os.listdir(file_root)
file = file_list[file_num]
print(file)
tree = parse(file_root + file)
root = tree.getroot()
# -




def normalize(node_arr, size = 511):
    node_arr = np.array(node_arr)
    min_x, max_x = min(node_arr[:,0]), max(node_arr[:,0])
    min_y, max_y = min(node_arr[:,1]), max(node_arr[:,1])
    max_length = max(max_x - min_x, max_y - min_y)
    node_arr[:,0] =  (node_arr[:,0] - min_x) * (size/max_length)
    node_arr[:,1] =  (node_arr[:,1] - min_y) * (size/max_length)
    return node_arr.tolist()


def fill_in_array(arr):
    x,y = arr[0]
    x_before,y_before = arr[1]    


def draw(maps, arr, types):
#     arr = sorted(arr)
    for i in range(1, len(arr)):
        x,y = arr[i]
        x_before,y_before = arr[i-1]
        if x-x_before != 0:
            w = (y-y_before)/(x-x_before)
        else:
            w = 0
        b = y - x*w
        while(True):
            maps[math.floor(x_before)][math.floor(y_before)] = \
                max(types, maps[math.floor(x_before)][math.floor(y_before)]) #중복될 경우 우선순위 높은 것을 type으로 삼겠다.
            if abs(x_before - x) > abs(y_before - y):
                if math.floor(x_before) == math.floor(x):
                    break
                x_before += (1 * 1 if x_before <= x else -1)
                y_before = x_before * w + b
                
            else:
                if math.floor(y_before) == math.floor(y):
                    break
                y_before += (1 * 1 if y_before < y else -1)
#                 print(w, y, y_before)
#                 if w == 0:
#                     if round(y_before) == round(y):
#                         break
                if w != 0:
                    x_before = (y_before - b) / w
#             print(x_before, x)
    return maps


# +
file_root = "dataset/INTERACTION-Dataset-DR-v1_1/maps/"
os.makedirs(file_root + "map_arr/", exist_ok=True)
file_list = os.listdir(file_root)
s1 = set()
s1_sub =set()
way_dict = []
map_size = 512
# types_dict = {
#     'pedestrian_marking' = 
#     'stop_line'
#     'solid_solid'
#     'solid'
#     'curbstone'
#     'road_border'
#     'line_thick'
#     'line_thin'
#     'guard_rail'
#     'virtual'
#     'traffic_sign'
#     'usR1-1'
#     'low'
#     'line_thick'
#     'line_thin'
#     'virtual'
#     'usR1-1'
#     'low'
#     'solid'
#     'road_border'
# }

types_dict = {'stop_line' : 90,
    'pedestrian_marking' : 80,
    'line_thick' : 70,
    'line_thin' : 60,
    'traffic_sign' : 50, 
    'curbstone' : 40,
    'guard_rail' : 30,
    'virtual' : 20,          #a non-physical lane boundary, intended mainly for intersections
    'road_border' : 10     #the end of the road
}

for file in file_list[1:]:
#     file = file_list[file_num]
    print(file)
    if 'xy' in file:
        tree = parse(file_root + file)
        root = tree.getroot()
        node = root.findall('node')
        way = root.findall('way')
        relation = root.findall('relation')
        node_arr = []
        for n in node:
            x = float(n.get('x'))
            y = float(n.get('y'))
            node_arr.append([x,y])
        node_arr = normalize(node_arr, map_size-1)
        way_dict = []
        for w in way:
            t_dict = {}
            nds = w.findall("nd")
            xy = []
            for nd in nds:
                num = int(nd.get('ref'))-1000
                xy.append(node_arr[num])
            t_dict['node'] = xy
            t_dict['tag'] = {"type" : None, "subtype" : None}
            for t in w.findall("tag"):
                t_dict['tag'][t.get('k')] = t.get('v')
        #         tag_dict = {"type" : None, "subtype" : None}
        #         tag_dict[t.get('k')] = t.get('v')
        #         t_dict['tag'].append(tag_dict)
            way_dict.append(t_dict)
        maps = np.zeros((map_size,map_size))
        
        for w in way_dict:
            maps = draw(maps, w['node'], types_dict[w['tag']['type']])
        with open(file_root + "map_arr/"+ file.split('.')[0] + ".json" , "w") as json_data:
            json.dump(maps.tolist(), json_data)



# +
# df = maps
# plt.pcolor(df)
# plt.colorbar()
# plt.show()




# +
# df = maps[0:100, 100:200]
# plt.pcolor(df)
# plt.colorbar()
# plt.show()



