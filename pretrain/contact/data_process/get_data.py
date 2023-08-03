import os
import re
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

residue_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
residue_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}

kind_dict={'H':26, 'L':27, 'P':28}


def get_element(line): # split by length
    aid=line[6:11].strip()
    atype=line[12:16].strip()
    rtype=line[17:20].strip()
    cname=line[21].strip()
    rid=line[22:26].strip()
    x=line[30:38].strip()
    y=line[38:46].strip()
    z=line[46:54].strip()
    return [aid,atype,rtype,cname,rid,x,y,z]

def residue_convert(arr):
    ret=[]
    for i in range(len(arr)):
        ret.append(residue_3to1[arr[i]])
    return np.array(ret)

def _find_range_idx(seq, target, mask):
    if pd.isna(target):
        return -1,-1
    slen,tlen=len(seq),len(target)

    i=0
    while i<=slen-tlen:
        j=0
        if seq[i]==target[j]:
            while (j<tlen) and (seq[i]==target[j]):
                i+=1
                j+=1
            if j==tlen:
                if mask[i-tlen]==0 and mask[i-1]==0:
                    return i-tlen,i-1 # index of the first and the last token
                else:
                    i-=j-1
                    continue
            else:
                i-=j-1
                continue
        else:
            i+=1
    return -1,-1

def residue_tokenize(seq):
    len_=len(seq)
    ret=[]
    for i in range(len_):
        ret.append(map_dict[seq[i]])
    return np.array(ret)

def get_sequence(file):
    f=open(os.path.join(root,file)+'.fasta','r',encoding='utf-8')
    flines=f.readlines()
    assert len(flines)==4
    
    assert flines[0].strip().split(':')[1]=='H'
    assert flines[2].strip().split(':')[1]=='L'
    
    return flines[1].strip(),flines[3].strip()

# not strict match endure
endure=5

save_name='oas'
root='../data/IG/raw/oas/predictions_flat'
files_all=os.listdir(root)

name_set_pdb=set()
name_set_fasta=set()
for file in files_all:
    if file[-3:]=='pdb':
        name_set_pdb.add(file[:-4])
    elif file[-5:]=='fasta':
        name_set_fasta.add(file[:-6])
#for a in name_set_pdb:
#    assert a in name_set_fasta

name_set_pdb=list(name_set_pdb)[:10]

data_all=[]
error_count=0

# not strict match bias
hbbias_count=[0,0,0,0,0]
hebias_count=[0,0,0,0,0]
lbbias_count=[0,0,0,0,0]
lebias_count=[0,0,0,0,0]

for i in tqdm(range(len(name_set_pdb))):
    file_id=name_set_pdb[i]
    
    f_pdb=open(os.path.join(root,file_id)+'.pdb','r',encoding='utf-8')
    lines=f_pdb.readlines()
    f_pdb.close()
    
    fhseq, flseq = get_sequence(file_id)
    
    
    data_=dict()
    data_['file']=file_id

    atoms=[]
    for line in lines:
        if line[:4]=='ATOM':
            atoms.append(get_element(line.strip())) #['0 atom_id','1 atom','2 res','3 chain','4 res_id','5 x','6 y','7 z','8 1.00','9 16.95','10 atom'] [0 aid, 1 atype, 2 rtype, 3 cname, 4 rid, 5 x, 6 y, 7 z]
    atoms=np.array(atoms)
    atoms=atoms[atoms[:,1]=='CA']
    

    #get hseq atoms
    hchain_atoms=atoms[atoms[:,3]=='H']
        
    all_residue=residue_convert(hchain_atoms[:,2])
    mask_=np.zeros(len(all_residue)).astype(bool)
        
    # not strict match
    for bbia_ in range(endure):
        for ebia_ in range(endure):
            tp_=fhseq[bbia_:len(fhseq)-ebia_]
            hbidx,heidx=_find_range_idx(all_residue,tp_,mask_)
            if hbidx>=0:
                df_hseq=tp_
                hbbias_count[bbia_]+=1
                hebias_count[ebia_]+=1
                if bbia_ or ebia_:
                    print('hseq, not strict match, bia1:{}, bia2:{}'.format(bbia_,ebia_))
                break
        if hbidx>=0:
            break
        
        
    if hbidx>=0:
        mask_[hbidx:heidx+1]=1
        hchain_atoms=hchain_atoms[mask_]
        hchain_atoms[:,2]=residue_convert(hchain_atoms[:,2])
        hchain_atoms[:,5]=hchain_atoms[:,5].astype(float)
        hchain_atoms[:,6]=hchain_atoms[:,6].astype(float)
        hchain_atoms[:,7]=hchain_atoms[:,7].astype(float)
        hseq="".join(hchain_atoms[:,2])
        assert hseq==fhseq
        data_['hseq']=hseq

        res_type=residue_tokenize(hseq)
        hdata=np.concatenate([res_type.reshape(-1,1),hchain_atoms[:,[5,6,7]].astype(np.float16)],axis=1)
        data_['hdata']=hdata
    else:
        data_['hseq']=None
        data_['hdata']=None
            
            
    #get lseq atoms
    lchain_atoms=atoms[atoms[:,3]=='L']
        
    all_residue=residue_convert(lchain_atoms[:,2])
    mask_=np.zeros(len(all_residue)).astype(bool)
        
    # not strict match
    for bbia_ in range(endure):
        for ebia_ in range(endure):
            tp_=flseq[bbia_:len(flseq)-ebia_]
            lbidx,leidx=_find_range_idx(all_residue,tp_,mask_)
            if lbidx>=0:
                df_lseq=tp_
                lbbias_count[bbia_]+=1
                lebias_count[ebia_]+=1
                if bbia_ or ebia_:
                    print('lseq, not strict match, bia1:{}, bia2:{}'.format(bbia_,ebia_))
                break
        if lbidx>=0:
            break
        
        
    if lbidx>=0:
        mask_[lbidx:leidx+1]=1
        lchain_atoms=lchain_atoms[mask_]
        lchain_atoms[:,2]=residue_convert(lchain_atoms[:,2])
        lchain_atoms[:,5]=lchain_atoms[:,5].astype(float)
        lchain_atoms[:,6]=lchain_atoms[:,6].astype(float)
        lchain_atoms[:,7]=lchain_atoms[:,7].astype(float)
        lseq="".join(lchain_atoms[:,2])
        assert lseq==flseq
        data_['lseq']=lseq

        res_type=residue_tokenize(lseq)
        ldata=np.concatenate([res_type.reshape(-1,1),lchain_atoms[:,[5,6,7]].astype(np.float16)],axis=1)
        data_['ldata']=ldata
    else:
        data_['lseq']=None
        data_['ldata']=None

        
    data_all.append(data_)
    
with open('{}_data.pickle'.format(save_name),'wb') as f:
    pickle.dump(data_all,f)
print('error count:',error_count)

print('hbbias_count',hbbias_count)
print('hebias_count',hebias_count)
print('lbbias_count',lbbias_count)
print('lebias_count',lebias_count)
