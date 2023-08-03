# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
import numpy as np
import mindspore.dataset as ds
import random
import mindspore as ms
from mindspore.mindrecord import FileWriter
import os
import argparse
import collections
import multiprocessing
import moxing as mox
import glob
import time
import gc
import pandas as pd
import re
import pickle

cv_schema_json = {"input_ids": {"type": "int32", "shape": [-1]}, # token id
                  "input_mask": {"type": "int32", "shape": [-1]}, # mask
                  "token_type_ids": {"type": "int32", "shape": [-1]}, # token type id
                  "cont_label": {"type":"int32", "shape": [-1]}, # label
                  "cont_label_weight": {"type":"float32", "shape": [-1]}, # label weight
                  "seq_len" : {"type":"int32"}
                  }

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}

kind_dict={'H':26, 'L':27, 'P':28}

# [KIND] P:pair H:heavy L:light
# [CLS] [KIND] ASEQ [SEP] GSEQ [SEP] [PAD]

# pair: [CLS] P H_ASEQ [SEP] L_ASEQ [SEP] H_GSEQ [SEP] L_GSEQ [SEP] [PAD] / [CLS] 0 H_ASEQ [SEP] H_GSEQ [SEP] L_ASEQ [SEP] L_GSEQ [SEP] [PAD]
# heavy: [CLS] H ASEQ [SEP] GSEQ [SEP] [PAD]
# light: [CLS] L ASEQ [SEP] GSEQ [SEP] [PAD]

def token_regular(seq):
    seq=re.sub('[BJOU\*]','X',seq)
    seq=re.sub('\-','',seq)
    return seq

def get_one_contact_map(data,thres=8):
    num=len(data)
    ret=np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            coor1=np.array(data[i][1:])
            coor2=np.array(data[j][1:])
            ret[i][j]=(np.sqrt(((coor1-coor2)**2).sum())<=thres)
            
    return ret


def get_attr(kind, line, max_sl, max_ml):
    seq_len = 110
    if line['{}score'.format(kind.lower())]<85:
        print('# low confidence')
        return False
    if len(line['{}seq'.format(kind.lower())])<seq_len:
        return False
    
    try:
        line['{}seq'.format(kind.lower())]=token_regular(line['{}seq'.format(kind.lower())])
        line['{}germ'.format(kind.lower())]=token_regular(line['{}germ'.format(kind.lower())])
    except:
        print('### in get_mask, token_regular error',line)
        return False
    
    seq_new = line['{}seq'.format(kind.lower())][:seq_len]
    germ_new = line['{}germ'.format(kind.lower())][:seq_len]
    seq_len = len(seq_new) #len(line['{}seq'.format(kind.lower())])
    seq = [map_dict['[CLS]']] + [kind_dict[kind]] + [map_dict[x] for x in seq_new] + \
          [map_dict['[SEP]']] + [map_dict[x] for x in germ_new] + [map_dict['[SEP]']]
    token_type = [0]*(2 + len(seq_new) + 1) + [1]*(len(germ_new) + 1)
    seq = np.array(seq)
    token_type = np.array(token_type)
    label = get_one_contact_map(line['{}data'.format(kind.lower())]).flatten()
    
    input_mask = np.pad(np.ones(seq.shape, dtype=np.int32), (0, max_sl - len(seq)),
                        constant_values=map_dict['[PAD]'])
    input_id = np.pad(seq.astype(np.int32), (0, max_sl - len(seq)),
                      constant_values=map_dict['[PAD]'])
    token_type = np.pad(token_type.astype(np.int32), (0, max_sl - len(token_type)),
                       constant_values=map_dict['[PAD]'])
    label = label[:max_ml] #最多预测个数

    label_weight = np.pad(np.ones(label.shape, dtype=np.float32), (0, max_ml - len(label)),
                              constant_values=map_dict['[PAD]'])
    label = np.pad(label.astype(np.int32), (0, max_ml - len(label)),
                          constant_values=map_dict['[PAD]'])
    return input_id, input_mask, token_type, label, label_weight, seq_len


def get_lines_pickle(dataset_file):
    lines = []
    for file in dataset_file:
        lines += pickle.load(open(file,'rb'))
    print("Total lines:", len(lines))
    return lines


def process(data_file, max_sl, max_ml, mask_prob, md_file_name, anc_prob, save_dir, kind):
    assert kind in ['H','L']
    md_name = os.path.join(save_dir, md_file_name + '.mindrecord')
    print(">>>>>>>>>>>>>>>>>save data:", md_name)
    writer = FileWriter(file_name=md_name, shard_num=1, overwrite=True)
    writer.add_schema(cv_schema_json, "train_schema")
    actual_length = FLAGS["max_seq_length"] - 2
    
    data = []
    if type(data_file)==list:
        lines = get_lines_pickle(data_file)
    else:
        lines = get_lines_pickle([data_file])
    random.shuffle(lines)
    
    count = 0
    for i, line in enumerate(lines):
        item = get_attr(kind = kind, #chain type
                        line = line,
                        max_sl=max_sl,
                        max_ml=max_ml)
        if not item:
            continue
        features = collections.OrderedDict()
        features["input_ids"] = item[0]
        features["input_mask"] = item[1]
        features["token_type_ids"] = item[2]
        features["cont_label"] = item[3]
        features["cont_label_weight"] = item[4]
        features["seq_len"] = item[5]
        
        
        data.append(features)
        count += 1
        if count == 10000:
            writer.write_raw_data(data)
            data = []
            count = 0
    if not len(data) == 0:
        writer.write_raw_data(data)
    writer.commit()


if __name__ == "__main__":
    
    DEBUG=True
    
    parser = argparse.ArgumentParser(description="Total data processing")
    parser.add_argument('--raw_data_path',
                        type=str,
                        default=None)
    parser.add_argument('--md_data_path',
                        type=str,
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=27)
    parser.add_argument('--masked_lm_prob',
                        type=float,
                        default=0.25)
    parser.add_argument('--ancestor_prob',
                        type=float,
                        default=0.25)
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=512)
    parser.add_argument('--kind',
                        type=str)
    args = parser.parse_args()
    FLAGS = {}

    FLAGS["max_seq_length"] = int(args.max_seq_length)
    FLAGS["max_predictions_per_seq"] = 12100
    FLAGS["random_seed"] = int(args.seed)
    FLAGS["dupe_factor"] = 1
    FLAGS["masked_lm_prob"] = float(args.masked_lm_prob)
    np.random.seed(FLAGS["random_seed"])
    
    src_url = args.raw_data_path
    save_obs_path = args.md_data_path
    anc_prob = args.ancestor_prob
    
    if src_url==None:
        data_path='../data'
    else:
        data_path = "cache/data"
        mox.file.copy_parallel(src_url=src_url,
                                dst_url=data_path)
    
    file = glob.glob(data_path+"/*.pickle") #[0]
    
    file_num=len(file)
    print('pickle file num:',file_num)
    
    if DEBUG:
        for i in range(file_num):
            save_dir = "../data/result/"
            os.makedirs(save_dir, exist_ok=True)

            file_name=file[i]
            process(data_file=[file_name], max_sl=FLAGS["max_seq_length"], max_ml=FLAGS["max_predictions_per_seq"],
                    mask_prob=FLAGS["masked_lm_prob"], md_file_name= 'test',#os.path.basename(file_name).split('.')[0], 
                    anc_prob=anc_prob, save_dir=save_dir, kind=args.kind)
    
    
'''    
    for i in range(file_num):
        save_dir = "cache/md_data_{}".format(i)
        os.makedirs(save_dir, exist_ok=True)
        
        file_name=file[i]
        process(data_file=[file_name], max_sl=FLAGS["max_seq_length"], max_ml=FLAGS["max_predictions_per_seq"],
                mask_prob=FLAGS["masked_lm_prob"], md_file_name=os.path.basename(file_name).split('.')[0], 
                anc_prob=anc_prob, save_dir=save_dir, kind=args.kind)
        
        print(">>>>>>>>>>>>>>>>>Begin to copy data")
        mox.file.copy_parallel(src_url=save_dir, dst_url=save_obs_path)
        print("Finish copy data")
'''