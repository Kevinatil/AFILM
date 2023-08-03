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

name='new'


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
        seq_set.add(json_data[i]['sequence'])
        tp_.append(json_data[i]['sequence'])
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
print(len(seq_set))


pd.DataFrame(data_all,columns=['sequence','cdr1','label1','cdr2','label2','cdr3','label3']).to_csv('../../../data/paratope/info_{}.csv'.format(name), 
                                                                                                   index=False)

def padding(seq,len_=200):
    return seq+'-'*(len_-len(seq))

def get_encoding(seqs):
    data=[]
    for i in range(len(seqs)):
        data.append(('seq{}'.format(i),padding(seqs[i])))

    ret=[]
    batch_size=4
    for idx in tqdm(range(0,len(seqs),batch_size)):
        data_=data[idx:idx+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]

        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[last_layer], return_contacts=True)
        token_representations = results["representations"][last_layer][:,1:200+1,:].cpu().data.numpy()

        ret.append(token_representations)

    ret=np.concatenate(ret,axis=0)
    return ret

if 0:
    f=open('../../../data/paratope/{}_all_sequence.txt'.format(name),'w')
    for seq in seq_set:
        #f.write(padding(seq)+'\n')
        f.write(seq+'\n')
    f.close()    
    
if 1:
    embed_all=dict()
    
    data_train=pickle.load(open('../../../data/paratope/new_all_sequence.txt_embedding.pickle','rb'))
    #data_train=np.max(data_train,axis=1)

    #data_train = get_encoding(seq_set)

    print(len(seq_set))
    print(len(data_train))

    for i in range(len(seq_set)):
        embed_all[seq_set[i]]=data_train[i]


    with open('../../../data/paratope/embed_all_{}.pickle'.format(name),'wb') as f:
        pickle.dump(embed_all,f)
