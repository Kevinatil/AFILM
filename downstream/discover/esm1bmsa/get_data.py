import pickle
import numpy as np
import pandas as pd

import random

k=10

name='esm1bmsa'

embed_all=pickle.load(open('../../../data/discover/embed_data_{}.pickle'.format(name),'rb'))
print(embed_all[list(embed_all.keys())[0]])
print(embed_all[list(embed_all.keys())[1]])

info=pd.read_csv('../../../data/discover/info_{}.csv'.format(name))
print(info.head())


## split train test set
#subjects=np.unique(info['subject'].values)

#train_sub=random.sample(subjects.tolist(),int(len(subjects)*0.82))
#train_sub=list(set(train_sub))
#test_sub=[]
#for sub in subjects:
#    if sub not in train_sub:
#        test_sub.append(sub)
#test_sub=list(set(test_sub))

#print(len(train_sub),len(test_sub))


train_sub=np.unique(info['subject'].values)

per_num=len(train_sub)//k+1
print(per_num)

def get_fold_data(subs):
    data=[]
    label=[]
    for i in range(len(subs)):
        sub_=subs[i]
        info_=info[info['subject']==sub_].values
        
        for j in range(len(info_)):
            data.append(embed_all[info_[j][0]])
            label.append(info_[j][1])
    
    return np.array(data),np.array(label)


random.shuffle(train_sub)

for i in range(k):
    tp_=train_sub[i*per_num:i*per_num+per_num]
    data_,label_=get_fold_data(tp_)
    print(data_.shape,label_.shape)
    np.save('../../../data/discover/{}_all_data_{}.npy'.format(name,i),data_)
    np.save('../../../data/discover/{}_all_label_{}.npy'.format(name,i),label_)
    np.save('../../../data/discover/{}_all_subject_{}.npy'.format(name,i),tp_)
    
    print(len(label_),label_.sum())

#tp_=test_sub
#data_,label_=get_fold_data(tp_)
#print(data_.shape,label_.shape)
#np.save('t5_test_data.npy',data_)
#np.save('t5_test_label.npy',label_)
#np.save('t5_test_subject.npy',tp_)

#print(len(label_),label_.sum())