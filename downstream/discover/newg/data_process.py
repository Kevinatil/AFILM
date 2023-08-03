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

#os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

label_map={'None':0,
           'SARS-COV-2':1}

name='newg'


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
germ_all=[]
disease_all=[]
subject_all=[]

for i in range(len(json_data)):
    tp_=json_data[i]
    seq_=seq_regular(tp_['sequence'])
    germ_=seq_regular(tp_['germline'])
    
    if seq_ in seq_set:
        continue
    else:
        seq_set.add(seq_)
        seq_all.append(seq_)
        germ_all.append(germ_)
        disease_all.append(label_map[tp_['Disease']])
        subject_all.append(tp_['Subject'])



print(len(seq_set))
print(len(seq_all))
print(len(set(seq_all)))
print(len(disease_all))
print(len(subject_all))

print(len(set(disease_all)))
print(len(set(subject_all)))

pd.DataFrame(np.array([seq_all,germ_all,disease_all,subject_all]).T,columns=
             ['sequence','germline','disease','subject']).to_csv('../../../data/discover/info_{}.csv'.format(name)
                                                      ,index=False)



df=pd.read_csv('../../../data/discover/info_{}.csv'.format(name))
seqs=df['sequence'].values
germs=df['germline'].values


def get_sequence(seqs,germs):
    f=open('../../../data/discover/{}_sequence.txt'.format(name),'w')
    for i in range(len(seqs)):
        seq=seqs[i]
        germ=germs[i]
        seq=re.sub(r'\-','X',seq)
        seq=re.sub(r'\*','X',seq)
        germ=re.sub(r'\-','X',germ)
        germ=re.sub(r'\*','X',germ)
        f.write(seq+','+germ+'\n')
    f.close()


if 0:
    get_sequence(seqs,germs)
    
    
if 1:
    data_all=pickle.load(open('../../../data/discover/{}_sequence.txt_embedding.pickle'.format(name),'rb'))

    embed_all=dict()

    for i in range(len(data_all)):
        embed_all[seqs[i]]= data_all[i][0,:]

    with open('../../../data/discover/embed_data_{}.pickle'.format(name),'wb') as f:
        pickle.dump(embed_all,f)