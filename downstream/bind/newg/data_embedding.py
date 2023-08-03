import os
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import json


name='newg'


def get_sequence(data, file_path):
    
    label_all=[]
    seq_all=[]
    germ_all=[]
    for i in range(len(data)):
        seq_all.append(data[i]['aligned_sequence'].upper())
        label_all.append(data[i]['label'])
        germ_all.append(data[i]['germline'].upper())
        
    f=open(file_path,'w')
    for i in range(len(seq_all)):
        f.write(seq_all[i]+','+germ_all[i]+'\n')
    f.close()
    
    label_all=np.array(label_all)
    print('(In embedding) embedding shape', label_all.shape)

    return label_all

json_train=[]
json_test=[]

# train file data
with open('../../../data/raw/mHER_H3.proc.adj.train.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_train.append(json.loads(json_str))
    
# val file data
with open('../../../data/raw/mHER_H3.proc.adj.valid.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_train.append(json.loads(json_str))
    
# test file data
with open('../../../data/raw/mHER_H3.proc.adj.test.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_test.append(json.loads(json_str))
    
print(len(json_train),len(json_test))


label_train=get_sequence(json_train, file_path='../../../data/bind/{}_train_sequence.txt'.format(name))
label_test=get_sequence(json_test, file_path='../../../data/bind/{}_test_sequence.txt'.format(name))


print(label_train.shape)
print(label_test.shape)

np.save('../../../data/bind/{}_train_label.npy'.format(name),label_train)
np.save('../../../data/bind/{}_test_label.npy'.format(name),label_test)