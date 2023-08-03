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


#pd.DataFrame(data_all,columns=['sequence','cdr1','label1','cdr2','label2','cdr3','label3']).to_csv('../../../data/paratope/info.csv', index=False)

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
            inputs=tokenizer.batch_encode_plus(seq_, add_special_tokens=True, padding='max_length', truncation=True, max_length=201)
            
            ids_=torch.tensor(inputs['input_ids']).to(device)
            mask_=torch.tensor(inputs['attention_mask']).to(device)
            word_embeddings_ = embed_backbone(ids_, mask_)[0]
            embedding_all.append(word_embeddings_.cpu().data.numpy())
    embedding_all=np.concatenate(embedding_all,axis=0)[:,1:,:]
    print('(In embedding) embedding shape',embedding_all.shape)
    print(embedding_all[0])
    
    return embedding_all


embed_all=dict()

backbone_name= 'BERT'
embed_backbone, tokenizer, embedding_size = create_model(backbone_name=backbone_name)
embed_backbone.to(device)

data_train = embedding_batch(seq_set, embed_backbone, tokenizer)

print(data_train.shape)

for i in range(len(seq_set)):
    embed_all[seq_set[i]]=data_train[i]


with open('../../../data/paratope/embed_all_bert.pickle','wb') as f:
    pickle.dump(embed_all,f)
