import pickle
import numpy as np
import pandas as pd

import random


name='newg'

k=10

embed_all=pickle.load(open('../../../data/paratope/embed_all_{}.pickle'.format(name),'rb'))
print(embed_all[list(embed_all.keys())[0]])
print(embed_all[list(embed_all.keys())[1]])

info=pd.read_csv('../../../data/paratope/info_{}.csv'.format(name))
print(info.head())


train_seq=info['sequence'].values.tolist()

per_num=len(train_seq)//k+1
print(per_num)

def get_label(seq, cdr, label, max_len=200): #12 345 6
    find_=seq.find(cdr)
    assert find_>=0
    
    return np.array([-1]*find_ + label + [-1]*(max_len-find_-len(label))).astype(float)

def get_fold_data(seqs, max_len=200):
    data=[]
    label=[]
    for i in range(len(seqs)):
        seq_=seqs[i]
        info_=info[info['sequence']==seq_].values[0]
        
        for j in range(3):
            tp_=embed_all[seq_][2:,:] # 去掉cls, kind
            pad_=np.zeros((max_len-tp_.shape[0],tp_.shape[1]))
            tp_=np.concatenate([tp_,pad_],axis=0)
            print(tp_.shape)
            data.append(tp_)
            cdr_=info_[1+2*j+1]
            label_=eval(info_[2+2*j+1])
            label.append(get_label(seq_,cdr_,label_,200))
    
    return np.array(data),np.array(label)



for i in range(k):
    tp_=train_seq[i*per_num:i*per_num+per_num]
    data_,label_=get_fold_data(tp_)
    print(data_.shape,label_.shape)
    np.save('../../../data/paratope/{}_train_data_{}.npy'.format(name,i),data_)
    np.save('../../../data/paratope/{}_train_label_{}.npy'.format(name,i),label_)
    np.save('../../../data/paratope/{}_train_seq_{}.npy'.format(name,i),tp_)
    

#tp_=test_seq
#data_,label_=get_fold_data(tp_)
#print(data_.shape,label_.shape)
#np.save('../../../data/paratope/t5_test_data.npy',data_)
#np.save('../../../data/paratope/t5_test_label.npy',label_)
#np.save('../../../data/paratope/t5_test_seq.npy',tp_)