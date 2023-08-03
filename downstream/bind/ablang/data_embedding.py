import os
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import json

import ablang

heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()

def get_encoding(seqs,model,mode,padding=False,len_=200):
    assert mode in ['seqcoding','rescoding']
    seqs_=[]
    if padding:
        for seq in seqs:
            seq=re.sub(r'\*','X',seq)
            seq+='-'*(len_-len(seq))
            seqs_.append(seq)
    else:
        for seq in seqs:
            seq=re.sub(r'\*','X',seq)
            seqs_.append(seq)
            
    batch_size=64
    
    embeds=[]
    
    for idx in tqdm(range(0,len(seqs_),batch_size)):
        seq_=seqs_[idx:idx+batch_size]
        embeds_=model(seq_,mode=mode)
        embeds.append(embeds_)
    
    
    return np.concatenate(embeds,axis=0)


def embedding_batch(data, shuffle=False):
    
    label_all=[]
    
    seq_all=[]
    for i in range(len(data)):
        seq_all.append(data[i]['aligned_sequence'].upper())
        label_all.append(data[i]['label'])
    
    embedding_all=get_encoding(seq_all,heavy_ablang,'seqcoding',False,-1)
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

np.save('../../../data/bind/ablang_train_data.npy',data_train)
np.save('../../../data/bind/ablang_train_label.npy',label_train)
np.save('../../../data/bind/ablang_test_data.npy',data_test)
np.save('../../../data/bind/ablang_test_label.npy',label_test)