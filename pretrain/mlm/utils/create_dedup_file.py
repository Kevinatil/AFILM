import os
import re
import gzip
import numpy as np
import pandas as pd

def text_regular(text,sub=r'\n'):
    return re.sub(sub,r'',text)

seq_type='heavy'

# 1 merge csv, csv to fasta
if 0:
    raw_dir='../../unpaired_{}/'.format(seq_type) # downloaded csv.tar.gz from OAS
    
    f=open('unpaired_{}.fasta'.format(seq_type),'a',encoding='utf-8')
    counter=0
    
    f_csv=open('unpaired_{}.csv'.format(seq_type),'a',encoding='utf-8')
    f_csv.write('sequence,germline,cdr1,cdr2,cdr3,species,file_name\n')
    
    files=os.listdir(raw_dir)
    for file in files:
        try:
            tp_=pd.read_csv(os.path.join(raw_dir,file),compression='gzip',sep=',',skiprows=[0])
        except:
            print('#SKIP {}'.format(file))
            continue
        aseq_=tp_['sequence_alignment_aa'].values
        gseq_=tp_['germline_alignment_aa'].values
        cdr1_=tp_['cdr1_aa'].values
        cdr2_=tp_['cdr2_aa'].values
        cdr3_=tp_['cdr3_aa'].values
        
        del tp_

        ##species
        f_tp=gzip.open(os.path.join(raw_dir,file),'rb')
        line=f_tp.readline().decode()
        f_tp.close()
        line=re.findall(r'""Species"": ""(.+?)"",',line)

        len_=len(aseq_)
        for i in range(len_):
            f.write('>{}\n{}\n'.format(counter,aseq_[i]))
            f_csv.write('{},{},{},{},{},{},{}\n'.format(aseq_[i],gseq_[i],cdr1_[i],cdr2_[i],cdr3_[i],line[0],file))

            counter+=1
        

    f.close()
    f_csv.close()


# 2 mmseqs cluster, create tsv


# 3 get rep idx from tsv
if 1:
        df=pd.read_csv('clusterRes_cluster.tsv'.format(i),'\t',header=None,names=['rep','seq'])
        rep_id=np.unique(df['rep'].values)
        print('rep_id',rep_id.shape)
        del df

        df=pd.read_csv('unpaired_heavy.csv'.format(i))
        seq_=df['sequence'][rep_id].values
        germ_=df['germline'][rep_id].values
        cdr1_=df['cdr1'][rep_id].values
        cdr2_=df['cdr2'][rep_id].values
        cdr3_=df['cdr3'][rep_id].values
        del df
        
        df=pd.DataFrame()
        df['sequence']=seq_
        df['germline']=germ_
        df['cdr1']=cdr1_
        df['cdr2']=cdr2_
        df['cdr3']=cdr3_
        df.to_csv('processed_data/uh.csv'.format(i),index=False)
        
