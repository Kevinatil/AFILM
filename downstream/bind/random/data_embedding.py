import os
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import json

name='random'

def get_random_encoding(seqs):
    ret=[]
    for seq in seqs:
        ret.append(np.random.randn(1,1024))
    return np.concatenate(ret,axis=0)


def embedding_batch(data, shuffle=False):
    
    label_all=[]
    
    seq_all=[]
    for i in range(len(data)):
        seq_all.append(data[i]['aligned_sequence'].upper())
        label_all.append(data[i]['label'])
    
    embedding_all=get_random_encoding(seq_all)
    label_all=np.array(label_all)
    #label_all=np.concatenate(label_all,axis=0)
    print('(In embedding) embedding shape',embedding_all.shape, label_all.shape)

    return embedding_all, label_all


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

data_train, label_train = embedding_batch(json_train)
data_test, label_test = embedding_batch(json_test)


print(data_train.shape)
print(label_train.shape)
print(data_test.shape)
print(label_test.shape)

np.save('../../../data/bind/{}_train_data.npy'.format(name),data_train)
np.save('../../../data/bind/{}_train_label.npy'.format(name),label_train)
np.save('../../../data/bind/{}_test_data.npy'.format(name),data_test)
np.save('../../../data/bind/{}_test_label.npy'.format(name),label_test)