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
             ['sequence','disease','subject']).to_csv('../../../data/discover/info_t5.csv'
                                                      ,index=False)



df=pd.read_csv('../../../data/discover/info_t5.csv')
seqs=df['sequence'].values



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
            embedding_all.append(word_embeddings_.cpu().data.numpy()[:,0,:])
    embedding_all=np.concatenate(embedding_all,axis=0)
    print('(In embedding) embedding shape',embedding_all.shape)
    print(embedding_all[0])
    
    return embedding_all




backbone_name= 'T5-XL-UNI' #'BERT'
embed_backbone, tokenizer, embedding_size = create_model(backbone_name=backbone_name)
embed_backbone.to(device)

data_all = embedding_batch(seqs, embed_backbone, tokenizer)

embed_all=dict()

for i in range(len(data_all)):
    embed_all[seqs[i]]=data_all[i]

with open('../../../data/discover/embed_data_t5.pickle','wb') as f:
    pickle.dump(embed_all,f)