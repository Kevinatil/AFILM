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

import ablang

#os.environ['CUDA_VISIBLE_DEVICES']='1,0,2,3'

name='ablang'

heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()
light_ablang = ablang.pretrained("light")
light_ablang.freeze()


json_data=[]

# file data
with open('../../../data/raw/Paratope.germline.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    json_data.append(json.loads(json_str))

print(len(json_data))




seq_set=set()
data_all=[]
heavy_num=0
tp_=[]
for i in range(len(json_data)):
    if i%3==0:
        if i%6==0:
            seq_set.add((json_data[i]['sequence'],'H'))
            heavy_num+=1
        elif i%6==3:
            seq_set.add((json_data[i]['sequence'],'L'))
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

print(heavy_num)

pd.DataFrame(data_all,columns=['sequence','cdr1','label1','cdr2','label2','cdr3','label3']).to_csv('../../../data/paratope/info_{}.csv'.format(name), 
                                                                                                   index=False)

def padding(seq,len_=200):
    return seq+'-'*(len_-len(seq))

def get_one_encoding(seqs,model,mode,padding=True,len_=200):
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
            
    embeds=model(seqs_,mode=mode)
    
    return np.stack(embeds,axis=0)


def get_encoding(seqs):
    ret=[]
    heavy_num=0
    for i in tqdm(range(len(seqs))):
        seq_,kind_=seqs[i]
        if kind_=='H':
            ret.append(get_one_encoding([seq_],heavy_ablang,'rescoding',True,200))
        else:
            ret.append(get_one_encoding([seq_],light_ablang,'rescoding',True,200))

    ret=np.concatenate(ret,axis=0)
    
    print('heavy',heavy_num)
    
    return ret


embed_all=dict()

#backbone_name= 'T5-XL-UNI' #'BERT'
#embed_backbone, tokenizer, embedding_size = create_model(backbone_name=backbone_name)
#embed_backbone.to(device)

data_train = get_encoding(seq_set)

print(len(seq_set))
print(data_train.shape)

for i in range(len(seq_set)):
    embed_all[seq_set[i][0]]=data_train[i]


with open('../../../data/paratope/embed_all_{}.pickle'.format(name),'wb') as f:
    pickle.dump(embed_all,f)
