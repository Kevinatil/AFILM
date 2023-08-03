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

os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

label_map={'None':0,
           'SARS-COV-2':1}

name='esm2'

last_layer=36

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
model.eval().cuda()
batch_converter = alphabet.get_batch_converter()


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
        #token_representations = results["representations"][12][:,1:seq_lens-1,:].cpu().data.numpy()
        token_representations = results["representations"][last_layer][:,0,:].cpu().data.numpy()

        ret.append(token_representations)

    ret=np.concatenate(ret,axis=0)
    return ret


data_all = get_encoding(seqs)

embed_all=dict()

for i in range(len(data_all)):
    embed_all[seqs[i]]=data_all[i]

with open('../../../data/discover/embed_data_{}.pickle'.format(name),'wb') as f:
    pickle.dump(embed_all,f)