import os
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import json

os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S")
model.eval().cuda()  # disables dropout for deterministic results
batch_converter = alphabet.get_batch_converter()

def get_encoding(seqs):
    data=[]
    for i in range(len(seqs)):
        data.append(('seq{}'.format(i),seqs[i]))

    ret=[]
    batch_size=64
    for idx in tqdm(range(0,len(seqs),batch_size)):
        data_=data[idx:idx+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]

        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[12], return_contacts=True)
        #token_representations = results["representations"][12][:,1:seq_lens-1,:].cpu().data.numpy()
        token_representations = results["representations"][12][0,:,0,:].cpu().data.numpy()

        ret.append(token_representations)

    ret=np.concatenate(ret,axis=0)
    return ret


def embedding_batch(data, shuffle=False):
    
    label_all=[]
    
    seq_all=[]
    for i in range(len(data)):
        seq_all.append(data[i]['aligned_sequence'].upper())
        label_all.append(data[i]['label'])
    
    embedding_all=get_encoding(seq_all)
    #label_all=np.concatenate(label_all,axis=0)
    label_all=np.array(label_all)
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

np.save('../../../data/bind/esm1bmsa_train_data.npy',data_train)
np.save('../../../data/bind/esm1bmsa_train_label.npy',label_train)
np.save('../../../data/bind/esm1bmsa_test_data.npy',data_test)
np.save('../../../data/bind/esm1bmsa_test_label.npy',label_test)