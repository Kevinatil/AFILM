import re
import pickle
import numpy as np

import pandas as pd
from tqdm import tqdm

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}
rmap_dict = {0:"[PAD]", 1:"[UNK]", 2:"[CLS]", 3:"[SEP]", 4:"[MASK]", 5:"L", 6:"A", 7:"G", 8:"V", 9:"E", 
             10:"S", 11:"I", 12:"K", 13:"R", 14:"D", 15:"T", 16:"P", 17:"N", 18:"Q", 19:"F", 20:"Y", 
             21:"M", 22:"H", 23:"C", 24:"W", 25:"X"}


def is_end(line):
    if len(line)==0:
        return False
    if line[0]=='V' or line[0]=='-' or line=='Alignments' or line[0]=='<' or line[:5]=='Query':
        return False
    return True

def process_one_item(lines):
    aligns=[]
    i=0
    while i<len(lines):
        line_=lines[i].strip()
        if len(line_)==0:
            i+=1
            continue
        if line_=='Alignments':
            while not is_end(line_):
                if len(line_) and (not ((line_[0]=='-') or (line_[0]=='<') or (line_[:5]=='Query') or (line_=='Alignments'))):
                    aligns.append(line_)
                i+=1
                line_=lines[i].strip()
        i+=1
    
    #print(aligns)
    
    name_set=set()
    for i in range(len(aligns)):
        find=re.findall(r'[V]  [0-9\.]+\% \(.+\)[\s]*(.+?)[\s]*[0-9]+[\s]*[A-Z\-\*]+  [0-9]+',aligns[i])[0]
        name_set.add(find)
    assert len(name_set)==3
    name_set=list(name_set)
    
    seq_dict={}
    score_dict={}
    
    for name_ in name_set:
        seq_dict[name_]=""
        score_dict[name_]=0
        
    for i in range(len(aligns)):
        find=re.findall(r'[V]  ([0-9\.]+)\% \(.+\)[\s]*(.+?)[\s]*[0-9]+[\s]*([A-Z\-\*]+)  [0-9]+',aligns[i])[0]
        seq_dict[find[1]]+=find[2]
        score_dict[find[1]]=float(find[0])
        
    ret=[]
        
    for name_ in name_set:
        ret.append([score_dict[name_],seq_dict[name_]])
    
        
    return ret

def get_germline_csv(species, chain):
    print(species,chain)
    f=open('../tools/_cache/output_{}_{}.txt'.format(species,chain),'r')
    #f=open('tp.txt'.format(species,chain),'r')
    file_lines=f.readlines()
    f.close()

    res_all=[]

    i=0
    while i < len(file_lines):
        line_=file_lines[i].strip()
        if len(line_)==0:
            i+=1
            continue

        if line_[:6]=='Query=':
            lines_tp=[]
            file_=re.findall(r'Query= (.+)',line_)[0]
            #print(file_)
            i+=1
            while i < len(file_lines) and (file_lines[i].strip()+'123456')[:6] != 'Query=':
                lines_tp.append(file_lines[i].strip())
                i+=1

            ret_=process_one_item(lines_tp)

            score_max=0
            seq=None
            for j_ in range(len(ret_)):
                score_=ret_[j_][0]
                seq_=ret_[j_][1]
                if score_>score_max:
                    seq=seq_
                    score_max=score_

            res_all.append([file_,score_max,seq])

        else:
            i+=1

    #pd.DataFrame(res_all,columns=['file','score','germline']).to_csv('{}_{}.csv'.format(species,chain),index=False)
    pd.DataFrame(res_all,columns=['file','score','germline']).to_csv('{}_{}_tp.csv'.format(species,chain),index=False)

# get germline
if 0:
    get_germline_csv('human','light')
    get_germline_csv('mouse','heavy')
    get_germline_csv('mouse','light')
    get_germline_csv('rat','heavy')
    get_germline_csv('rat','light')

# get the highest score
if 1:
    
    data_germ=[]
    
    df_h_human=pd.read_csv('human_heavy.csv').values
    df_l_human=pd.read_csv('human_light.csv').values
    df_h_mouse=pd.read_csv('mouse_heavy.csv').values
    df_l_mouse=pd.read_csv('mouse_light.csv').values
    df_h_rat=pd.read_csv('rat_heavy.csv').values
    df_l_rat=pd.read_csv('rat_light.csv').values
    
    h_df=[df_h_human,df_h_mouse,df_h_rat]
    l_df=[df_l_human,df_l_mouse,df_l_rat]
    
    data=pickle.load(open('oas_data.pickle','rb'))
    
    for i in tqdm(range(len(data))):
    #for i in tqdm(range(10)):
    
        data_=data[i]
    
        file=data[i]['file']
        
        #hgerm
        germ_=None
        score_=0
        for df_ in h_df:
            tp_=df_[df_[:,0]==file][0]
            if tp_[1]>score_:
                score_=tp_[1]
                germ_=tp_[2]
        
        data_['hgerm']=germ_
        data_['hscore']=score_
        
        
        #lgerm
        germ_=None
        score_=0
        for df_ in l_df:
            tp_=df_[df_[:,0]==file][0]
            if tp_[1]>score_:
                score_=tp_[1]
                germ_=tp_[2]
        
        data_['lgerm']=germ_
        data_['lscore']=score_
        
        data_germ.append(data_)
        
    with open('oas_data_germ.pickle','wb') as f:
        pickle.dump(data_germ,f)


if 0:
    f=open('../tools/_cache/output_human_heavy.txt','r')
    file_lines=f.readlines()
    f.close()
    
    f=open('tp.txt','w')
    for i in range(500):
        f.write(file_lines[i])
    f.close()
    

