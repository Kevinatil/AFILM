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

cv_schema_json = {"input_ids": {"type": "int32", "shape": [-1]}, #MLM，token id
                  "input_mask": {"type": "int32", "shape": [-1]}, #MLM，mask
                  "masked_lm_positions": {"type": "int32", "shape": [-1]}, #MLM position
                  "masked_lm_ids": {"type": "int32", "shape": [-1]}, #MLM label
                  "masked_lm_weights": {"type": "float32", "shape": [-1]}, #MLM
                  
                  "token_type_ids": {"type": "int32", "shape": [-1]}
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
    return re.sub('[BJOU\*]','X',seq)

def get_focused_span_mask(seq, cdr_dict, mask_prob, focus_range, max_ml):
    ismask=np.zeros(seq.shape)
    #print(cdr_dict)
    # mask cdr
    masked_len=0
    for range_ in focus_range:
        bidx,eidx=_find_range_idx(seq,cdr_dict[range_])
        if bidx>=0:
            len_=eidx-bidx
            bidx+=random.randint(0,len_//3) # mask part of cdr
            eidx-=random.randint(0,len_//3)
            ismask[bidx:eidx+1]=1
            masked_len+=eidx+1-bidx
            
    # mask rest
    mask_len = min(len(seq) * mask_prob, max_ml) - masked_len
    sidxs = iter(np.random.permutation(len(seq)))
    for trial in range(3):
        slens = np.random.poisson(3, len(seq))
        slens[slens < 2] = 2
        slens[slens > 8] = 8
        slens = slens[slens.cumsum() < mask_len]
        if len(slens) != 0:
            break
    for slen in slens:
        for trial in range(3):
            sid = next(sidxs)
            lid = sid - 1      # do not merge two spans
            rid = sid + slen   # do not merge two spans
            if lid >= 0 and rid < len(seq) and ismask[lid] != 1 and ismask[rid] != 1:
                ismask[sid: sid + slen] = 1
                break
    
    ismask*=_legal_mask(seq)
    
    return ismask


def numpy_mask_tokens(seq, cdr_dict, mask_prob, focus_range, max_ml):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = np.copy(seq)
    
    masked_indices = get_focused_span_mask(seq, cdr_dict, mask_prob, focus_range, max_ml).astype(bool)
    
    masked_lm_positions = np.where(masked_indices == True)[0]

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
    seq[indices_replaced] = map_dict['[MASK]']

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced)
    random_words = np.random.randint(
        low=5, high=25, size=np.count_nonzero(indices_random), dtype=np.int64
    )
    seq[indices_random] = random_words
    masked_lm_ids = labels[masked_lm_positions]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    
    #print(seq, labels)
    
    return seq, masked_lm_positions, masked_lm_ids

def _legal_mask(seq):
    return (seq>=5)*(seq<=25)

def _find_range_idx(seq, target):
    slen,tlen=len(seq),len(target)
    if tlen==0:
        return -1,-1

    i=0
    while i<=slen-tlen:
        j=0
        if seq[i]==target[j]:
            while (j<tlen) and (seq[i]==target[j]):
                i+=1
                j+=1
            if j==tlen:
                return i-tlen,i-1 # index of the first and the last token
            else:
                i-=j-1
                continue
        else:
            i+=1
    return -1,-1


def get_mask(kind, line, max_sl, max_ml, mask_prob, focus_range=['cdr1','cdr2','cdr3']):
    # line: [sequence,germline,cdr1,cdr2,cdr3]
    try:
        for i in range(len(line)):
            if not pd.isna(line[i]):
                line[i]=token_regular(line[i])
    except:
        print('### in get_mask, token_regular error',line)
        return False
    
    
    seq = [map_dict['[CLS]']] + [kind] + [map_dict[x] for x in line[0]] + [map_dict['[SEP]']] + [map_dict[x] for x in line[1]] + [map_dict['[SEP]']]
    token_type = [0]*(2 + len(line[0]) + 1) + [1]*(len(line[1]) + 1)
    seq = np.array(seq)
    token_type = np.array(token_type)
    
    all_focus_range=['cdr1','cdr2','cdr3']
    cdr_dict={}
    for i_ in range(len(all_focus_range)):
        ran_=all_focus_range[i_]
        if not pd.isna(line[i_+2]):
            cdr_dict[ran_]=[map_dict[x] for x in line[i_+2]]
        else:
            cdr_dict[ran_]=[]

    #cdr_dict={'cdr1':[map_dict[x] for x in line[2]], 
    #          'cdr2':[map_dict[x] for x in line[3]],
    #          'cdr3':[map_dict[x] for x in line[4]]}
    
    input_id, masked_lm_position, masked_lm_id = numpy_mask_tokens(seq=seq, 
                                                                   cdr_dict = cdr_dict,
                                                                   mask_prob = mask_prob,
                                                                   focus_range = focus_range,
                                                                   max_ml = max_ml)
    input_mask = np.pad(np.ones(input_id.shape, dtype=np.int32), (0, max_sl - len(seq)),
                        constant_values=map_dict['[PAD]'])
    input_id = np.pad(input_id.astype(np.int32), (0, max_sl - len(seq)),
                      constant_values=map_dict['[PAD]'])
    token_type = np.pad(token_type.astype(np.int32), (0, max_sl - len(token_type)),
                       constant_values=map_dict['[PAD]'])
    masked_lm_id = masked_lm_id[:max_ml] #最多预测个数
    masked_lm_position = masked_lm_position[:max_ml]
    masked_lm_weight = np.pad(np.ones(masked_lm_id.shape, dtype=np.float32), (0, max_ml - len(masked_lm_id)),
                              constant_values=map_dict['[PAD]'])
    masked_lm_position = np.pad(masked_lm_position.astype(np.int32), (0, max_ml - len(masked_lm_position)),
                                constant_values=map_dict['[PAD]'])
    masked_lm_id = np.pad(masked_lm_id.astype(np.int32), (0, max_ml - len(masked_lm_id)),
                          constant_values=map_dict['[PAD]'])
    return input_id, input_mask, masked_lm_position, masked_lm_id, masked_lm_weight, token_type


def get_lines(dataset_file):
    lines = []
    for txt in dataset_file:
        with open(txt, "r") as f:
            for line in f:
                if not len(line.strip()) <= 10:
                    lines.append(line.strip())
    print("Total lines:", len(lines))
    return lines

def get_lines_csv(dataset_file):
    lines = []
    column_order=['sequence','germline','cdr1','cdr2','cdr3']
    for file in dataset_file:
        data=pd.read_csv(file)
        if (data.columns!=column_order).any():
            data=data[column_order]
        data=data.values
        #data=[data.values[0]]
        for i in range(len(data)):
            lines.append(data[i])
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
        lines = get_lines_csv(data_file)
    else:
        lines = get_lines_csv([data_file])
    random.shuffle(lines)
    
    anc_lines_num=len(lines)
    
    count = 0
    for i, line in enumerate(lines):
        if pd.isna(line[1]):
            continue
        
        #mlm
        item = get_mask(kind = kind_dict[kind], #chain type
                        line = line, # [sequence,germline,cdr1,cdr2,cdr3]
                        max_sl=max_sl,
                        max_ml=max_ml,
                        mask_prob=mask_prob,
                        focus_range=['cdr1','cdr2','cdr3'])
        if not item:
            continue
        features = collections.OrderedDict()
        features["input_ids"] = item[0]
        features["input_mask"] = item[1]
        features["masked_lm_positions"] = item[2]
        features["masked_lm_ids"] = item[3]
        features["masked_lm_weights"] = item[4]
        features["token_type_ids"] = item[5]
        
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
    FLAGS["max_predictions_per_seq"] = 160
    FLAGS["random_seed"] = int(args.seed)
    FLAGS["dupe_factor"] = 1
    FLAGS["masked_lm_prob"] = float(args.masked_lm_prob)
    np.random.seed(FLAGS["random_seed"])
    src_url = args.raw_data_path
    save_obs_path = args.md_data_path
    anc_prob = args.ancestor_prob

    if not DEBUG:
        data_path = "cache/data"
        mox.file.copy_parallel(src_url=src_url,
                            dst_url=data_path)
        file = glob.glob("cache/data/*.csv") #[0]
    else:
        file = glob.glob("../../../../data/csv/debug/split/*.csv")

    
    file_num=len(file)
    print('csv file num:',file_num)
    
    
    if not DEBUG:
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
    else:
        for i in range(file_num):
            save_dir = "../../../../data/md/newds"
            os.makedirs(save_dir, exist_ok=True)

            file_name=file[i]
            process(data_file=[file_name], max_sl=FLAGS["max_seq_length"], max_ml=FLAGS["max_predictions_per_seq"],
                    mask_prob=FLAGS["masked_lm_prob"], md_file_name=os.path.basename(file_name).split('.')[0], 
                    anc_prob=anc_prob, save_dir=save_dir, kind=args.kind)