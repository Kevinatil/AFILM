import os
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import json

#os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

label_map={'immature_b_cell':0, 
           'transitional_b_cell':1, 
           'mature_b_cell':2,
           'plasmacytes_PC':3,
           'memory_IgD-':4,
           'memory_IgD+':5}

name='random'



def get_random_encoding(seqs):
    ret=[]
    for seq in seqs:
        ret.append(np.random.randn(1,1024))
    return np.concatenate(ret,axis=0)


json_data=[]

# file data
with open('../../../data/raw/Bcell.germline.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_data.append(json.loads(json_str))

print(len(json_data))

seq_set=set()
seq_all=[]
label_all=[]

for i in range(len(json_data)):
    seq_=json_data[i]['sequence']
    if seq_ in seq_set:
        continue
    else:
        seq_set.add(seq_)
        seq_all.append(seq_)
        label_all.append(label_map[json_data[i]['label']])

print(len(seq_set))
print(len(seq_all))
print(len(set(seq_all)))


data_all = get_random_encoding(seq_all)
label_all=np.array(label_all)

print(data_all.shape)
print(label_all.shape)

np.save('../../../data/bcell/{}_all_data_cls.npy'.format(name),data_all)
np.save('../../../data/bcell/{}_all_label.npy'.format(name),label_all)