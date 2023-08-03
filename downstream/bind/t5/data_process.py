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
#print(json_train[0])
#print(json_test[0])


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
        seq = self.jsons[index]['aligned_sequence'].upper()
        seq = " ".join("".join(seq.split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        label = self.jsons[index]['label']
        return seq, label
        
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
    
    BATCH_SIZE = 1024
    print('(In embedding) Loaded',backbone_name)
    
    Embed_dataset=SeqDataset(data)
    embed_loader = DataLoader(dataset=Embed_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    
    embedding_all=[]
    label_all=[]
    
    print('(In embedding) Begin embedding')
    embed_backbone.eval()
    with torch.no_grad():
        for step, (seq_, label_) in tqdm(enumerate(embed_loader)):
            inputs=tokenizer.batch_encode_plus(seq_, add_special_tokens=True, padding=True, truncation=True,max_length= 500)
            
            ids_=torch.tensor(inputs['input_ids']).to(device)
            mask_=torch.tensor(inputs['attention_mask']).to(device)
            word_embeddings_ = embed_backbone(ids_, mask_)[0]
            embedding_all.append(word_embeddings_.cpu().data.numpy())
            label_all.append(label_)
    embedding_all=np.concatenate(embedding_all,axis=0)
    label_all=np.concatenate(label_all,axis=0)
    print('(In embedding) embedding shape',embedding_all.shape, label_all.shape)
    print(embedding_all[0])
    print(label_all[0])
    
    return embedding_all, label_all

'''
backbone_name='T5-XL-UNI'
embed_backbone, tokenizer, embedding_size = create_model(backbone_name=backbone_name)
embed_backbone.to(device)


#data_train, label_train = embedding_batch(json_train)
data_test, label_test = embedding_batch(json_test, embed_backbone, tokenizer)

#np.save('../../../data/bind/t5_train_data.npy',data_train)
#np.save('../../../data/bind/t5_train_label.npy',label_train)
np.save('../../../data/bind/t5_test_data.npy',data_test)
np.save('../../../data/bind/t5_test_label.npy',label_test)

'''


train_data=np.load('../../../data/bind/t5_train_data.npy')
#train_data=np.max(train_data,axis=1)
np.save('../../../data/bind/t5_train_data_cls.npy',train_data[:,0,:])

test_data=np.load('../../../data/bind/t5_test_data.npy')
#test_data=np.max(test_data,axis=1)
np.save('../../../data/bind/t5_test_data_cls.npy',test_data[:,0,:])