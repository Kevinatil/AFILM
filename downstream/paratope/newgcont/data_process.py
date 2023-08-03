import os
import re
import json

import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizerFast, BertForPreTraining, BertTokenizer, BertModel,T5Tokenizer,T5EncoderModel
import pickle

import numpy as np
import time
import pandas as pd
from tqdm import tqdm

#os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

name='newgcont'


json_data=[]

# file data
with open('../../../data/raw/Paratope.germline.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_data.append(json.loads(json_str))

print(len(json_data))

seq_set=set()
data_all=[]

tp_=[]
for i in range(len(json_data)):
    if i%3==0:
        seq_set.add(json_data[i]['sequence']+','+json_data[i]['gemline'])
        tp_.append(json_data[i]['sequence'])
        tp_.append(json_data[i]['gemline'])
        tp_.append(json_data[i]['cdrs'])
        tp_.append(json_data[i]['label'])
    elif i%3==1:
        tp_.append(json_data[i]['cdrs'])
        tp_.append(json_data[i]['label'])
    else:
        tp_.append(json_data[i]['cdrs'])
        tp_.append(json_data[i]['label'])
        data_all.append(tp_)
        tp_=[]



seq_set=list(seq_set)
seq_set.sort()
print(len(seq_set))

print(seq_set[:5])


pd.DataFrame(data_all,columns=['sequence','germline','cdr1','label1','cdr2','label2','cdr3','label3']).to_csv('../../../data/paratope/info_{}.csv'.format(name), 
                                                                                                   index=False)

def padding(seq,len_=200):
    return seq+'-'*(len_-len(seq))


if 0:
    f=open('../../../data/paratope/{}_all_sequence.txt'.format(name),'w')
    for seq in seq_set:
        #f.write(padding(seq)+'\n')
        f.write(seq+'\n')
    f.close()
    
if 1:
    embed_all=dict()
    
    data_train=pickle.load(open('../../../data/paratope/{}_all_sequence.txt_embedding.pickle'.format(name),'rb'))
    #data_train=np.max(data_train,axis=1)

    print(len(seq_set))
    print(len(data_train))

    for i in range(len(seq_set)):
        embed_all[seq_set[i].split(',')[0]]=data_train[i]


    with open('../../../data/paratope/embed_all_{}.pickle'.format(name),'wb') as f:
        pickle.dump(embed_all,f)
