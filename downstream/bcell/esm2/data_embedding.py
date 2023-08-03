import os
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import json

os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

label_map={'immature_b_cell':0, 
           'transitional_b_cell':1, 
           'mature_b_cell':2,
           'plasmacytes_PC':3,
           'memory_IgD-':4,
           'memory_IgD+':5}

name='esm2'

last_layer=36

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
model.eval().cuda()
batch_converter = alphabet.get_batch_converter()

def get_encoding(seqs):
    data=[]
    for i in range(len(seqs)):
        seq_=re.sub(r'\*','X',seqs[i])
        data.append(('seq{}'.format(i),seq_))

    ret=[]
    batch_size=16
    for idx in tqdm(range(0,len(seqs),batch_size)):
        data_=data[idx:idx+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]

        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[last_layer], return_contacts=True)
        token_representations = results["representations"][last_layer][:,0,:].cpu().data.numpy()

        ret.append(token_representations)

    ret=np.concatenate(ret,axis=0)
    return ret


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


data_all = get_encoding(seq_all)
label_all=np.array(label_all)

print(data_all.shape)
print(label_all.shape)

np.save('../../../data/bcell/{}_all_data_cls.npy'.format(name),data_all)
np.save('../../../data/bcell/{}_all_label.npy'.format(name),label_all)