# +
import os
import glob
import json
from tqdm import tqdm
import numpy as np



paths = ["dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_EP0/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_EP1/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_CHN_Merging_ZS/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_CHN_Roundabout_LN/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_DEU_Merging_MT/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_DEU_Roundabout_OF/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_GL/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Roundabout_EP/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Roundabout_FT/",
"dataset/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Roundabout_SR/"]
# -

for path in tqdm(paths):
    save_path = path + "TE_AVG/"
    os.makedirs(save_path, exist_ok=True)
    csv_files = sorted(glob.glob(path + "vehicle*.csv"), key = lambda a: int(a.split('_')[-1].split('.')[0]))
    for f_i, csv_file in enumerate(csv_files): # select file
        file_name = csv_file.split(".")[0].split("/")[-1]
        json_files = glob.glob(path + "TE/" + file_name + "*")
        write_dict = {}
        for json_file in json_files:
            with open(json_file, "r") as json_data:
                data = json.load(json_data)
            try:
                for k in write_dict['TEmatrix'].keys():
                    write_dict['TEmatrix'][k] += np.array(data['TEmatrix'][k])
                    
            except:
                write_dict['data'] = data['data']
                write_dict['scene_attendent'] = data['scene_attendent']
                write_dict['TEmatrix'] = {}
                for k in data['TEmatrix'].keys():
                    write_dict['TEmatrix'][k] = np.array(data['TEmatrix'][k])
                    
        if len(json_files) == 10:
            for k in write_dict['TEmatrix'].keys():
                write_dict['TEmatrix'][k] /= len(json_files)
                write_dict['TEmatrix'][k] = write_dict['TEmatrix'][k].tolist()
                
            with open(save_path + file_name + ".json", "w") as json_data:
                json.dump(write_dict, json_data)
#         break
#     break

a = [[1,2],[3,4]]
b = [[1,2],[3,4]]
a+b


import numpy as np
a = np.array(a)


