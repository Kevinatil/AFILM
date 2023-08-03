import pickle
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}
rmap_dict = {0:"[PAD]", 1:"[UNK]", 2:"[CLS]", 3:"[SEP]", 4:"[MASK]", 5:"L", 6:"A", 7:"G", 8:"V", 9:"E", 
             10:"S", 11:"I", 12:"K", 13:"R", 14:"D", 15:"T", 16:"P", 17:"N", 18:"Q", 19:"F", 20:"Y", 
             21:"M", 22:"H", 23:"C", 24:"W", 25:"X"}

datas=pickle.load(open('oas_data.pickle','rb'))
print(len(datas))

def check_one(data):
    hseq_=data['hseq']
    data_=data['hdata'][:,0]
    seq=''
    for i in range(len(data_)):
        seq+=rmap_dict[int(data_[i])]
    assert seq==hseq_
    
    lseq_=data['lseq']
    data_=data['ldata'][:,0]
    seq=''
    for i in range(len(data_)):
        seq+=rmap_dict[int(data_[i])]
    assert seq==lseq_

    return len(hseq_),len(lseq_)
    
hseq_len=[]
lseq_len=[]

for i in tqdm(range(len(datas))):
    hseq_,lseq_=check_one(datas[i])
    hseq_len.append(hseq_)
    lseq_len.append(lseq_)
    
plt.hist(hseq_len)
plt.savefig('hseq_len.png')

plt.close()
plt.hist(lseq_len)
plt.savefig('lseq_len.png')

print(np.histogram(hseq_len))
print(np.histogram(lseq_len))