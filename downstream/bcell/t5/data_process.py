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


label_map={'immature_b_cell':0, 
           'transitional_b_cell':1, 
           'mature_b_cell':2,
           'plasmacytes_PC':3,
           'memory_IgD-':4,
           'memory_IgD+':5}

json_data=[]

# file data
with open('../../data/raw/Bcell.germline.jsonl', 'r') as json_file:
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


# encoding
device=torch.device('cuda')

def seq2token(data):
    data=data.upper()
    data = " ".join("".join(data.split()))
    data = re.sub(r"[UZOB]", "X", data)
    return data

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self,jsons):
        self.jsons = jsons

    def __getitem__(self, index):
        seq = self.jsons[index].upper()
        seq = " ".join("".join(seq.split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        return seq
        
    def __len__(self):
        return len(self.jsons)
    
def create_model(backbone_name="Rostlab/prot_bert"):
    path = "/userhome/liuxd/protTrans_classifier/pretrained_model" #"/userhome/xufan/sars_cov2_mutation/GISAID/0411/pretrained_model"
    if backbone_name == "BERT":
        pretrained_name = "Rostlab/prot_bert"
        pretrained_path = os.path.join(path, pretrained_name)
        tokenizer = BertTokenizer.from_pretrained(pretrained_name, do_lower_case=False)
        embedding_size = 1024 * 4
        try:
            embed_backbone = BertModel.from_pretrained(pretrained_path)
        except:
            embed_backbone = BertModel.from_pretrained(pretrained_name)
            embed_backbone.save_pretrained(pretrained_path)
    elif backbone_name == 'T5-BASE':
        pretrained_name = "Rostlab/prot_t5_base_mt_uniref50"
        pretrained_path = os.path.join(path, pretrained_name)
        tokenizer = T5Tokenizer.from_pretrained(pretrained_name, do_lower_case=False)
        embedding_size = 768 * 4
        try:
            embed_backbone = T5EncoderModel.from_pretrained(pretrained_path)
        except:
            embed_backbone = T5EncoderModel.from_pretrained(pretrained_name)
            embed_backbone.save_pretrained(pretrained_path)
    elif backbone_name == 'T5-XL-UNI':
        pretrained_name = "Rostlab/prot_t5_xl_uniref50"
        pretrained_path = os.path.join(path, pretrained_name)
        tokenizer = T5Tokenizer.from_pretrained(pretrained_name, do_lower_case=False)
        embedding_size = 1024 * 4
        try:
            print('in try, pretrained_path',pretrained_path)
            embed_backbone = T5EncoderModel.from_pretrained(pretrained_path)
        except:
            embed_backbone = T5EncoderModel.from_pretrained(pretrained_name)
            embed_backbone.save_pretrained(pretrained_path)
    return embed_backbone, tokenizer, embedding_size
    

def embedding_batch(data, embed_backbone, tokenizer, shuffle=False):
    
    BATCH_SIZE = 256
    print('(In embedding) Loaded',backbone_name)
    
    Embed_dataset=SeqDataset(data)
    embed_loader = DataLoader(dataset=Embed_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    
    embedding_all=[]
    
    print('(In embedding) Begin embedding')
    embed_backbone.eval()
    with torch.no_grad():
        for step, (seq_) in tqdm(enumerate(embed_loader)):
            inputs=tokenizer.batch_encode_plus(seq_, add_special_tokens=True, padding='max_length', truncation=True, max_length=200)
            
            ids_=torch.tensor(inputs['input_ids']).to(device)
            mask_=torch.tensor(inputs['attention_mask']).to(device)
            word_embeddings_ = embed_backbone(ids_, mask_)[0]
            word_embeddings_=word_embeddings_.cpu().data.numpy()
            
            #word_embeddings_=np.max(word_embeddings_,axis=1)
            embedding_all.append(word_embeddings_[:,0,:]) # cls token
    embedding_all=np.concatenate(embedding_all,axis=0)
    print('(In embedding) embedding shape',embedding_all.shape)
    #print(embedding_all[0])
    
    return embedding_all


backbone_name= 'T5-XL-UNI' #'BERT'
embed_backbone, tokenizer, embedding_size = create_model(backbone_name=backbone_name)
embed_backbone.to(device)

data_all = embedding_batch(seq_all, embed_backbone, tokenizer)
label_all=np.array(label_all)

print(data_all.shape)
print(label_all.shape)

np.save('../../data/bcell/t5_all_data_cls.npy',data_all)
np.save('../../data/bcell/t5_all_label.npy',label_all)

'''
X_train,X_test,y_train,y_test=train_test_split(data_all, label_all, train_size=0.8, random_state=27)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

np.save('../../data/bcell/t5_train_data.npy',X_train)
np.save('../../data/bcell/t5_test_data.npy',X_test)
np.save('../../data/bcell/t5_train_label.npy',y_train)
np.save('../../data/bcell/t5_test_label.npy',y_test)
'''