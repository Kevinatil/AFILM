import os
import re
import json

import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizerFast, BertForPreTraining, BertTokenizer, BertModel,T5Tokenizer,T5EncoderModel
import pickle

import random
import numpy as np
import time
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import ablang

#os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

label_map={'None':0,
           'SARS-COV-2':1}

name='ablang'

heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()


json_data=[]

# file data
with open('../../../data/raw/Sars.germline.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_data.append(json.loads(json_str))

print(len(json_data))

def seq_regular(seq):
    return re.sub('-','',seq)

seq_set=set()
seq_all=[]
disease_all=[]
subject_all=[]

for i in range(len(json_data)):
    tp_=json_data[i]
    seq_=seq_regular(tp_['sequence'])
    
    if seq_ in seq_set:
        continue
    else:
        seq_set.add(seq_)
        seq_all.append(seq_)
        disease_all.append(label_map[tp_['Disease']])
        subject_all.append(tp_['Subject'])



print(len(seq_set))
print(len(seq_all))
print(len(set(seq_all)))
print(len(disease_all))
print(len(subject_all))

print(len(set(disease_all)))
print(len(set(subject_all)))

pd.DataFrame(np.array([seq_all,disease_all,subject_all]).T,columns=
             ['sequence','disease','subject']).to_csv('../../../data/discover/info_{}.csv'.format(name)
                                                      ,index=False)



df=pd.read_csv('../../../data/discover/info_{}.csv'.format(name))
seqs=df['sequence'].values

def get_encoding(seqs,model,mode,padding=False,len_=200):
    assert mode in ['seqcoding','rescoding']
    seqs_=[]
    if padding:
        for seq in seqs:
            seq=re.sub('X','*',seq)
            seq+='-'*(len_-len(seq))
            seqs_.append(seq)
    else:
        for seq in seqs:
            seq=re.sub('X','*',seq)
            seqs_.append(seq)
            
    batch_size=64
    
    embeds=[]
    
    for idx in tqdm(range(0,len(seqs_),batch_size)):
        seq_=seqs_[idx:idx+batch_size]
        embeds_=model(seq_,mode=mode)
        embeds.append(embeds_)
    
    
    return np.concatenate(embeds,axis=0)



data_all = get_encoding(seqs,heavy_ablang,'seqcoding',False)

embed_all=dict()

for i in range(len(data_all)):
    embed_all[seqs[i]]=data_all[i]

with open('../../../data/discover/embed_data_{}.pickle'.format(name),'wb') as f:
    pickle.dump(embed_all,f)