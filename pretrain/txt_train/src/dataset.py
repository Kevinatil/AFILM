# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in run_pretrain.py
"""
import os
import math
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger
import pandas as pd
import re
import random

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
                    'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
                    'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}

kind_dict={'H':26, 'L':27, 'P':28}


def create_bert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None, batch_size=32,
                        bucket_list=None, dataset_format="mindrecord", num_samples=None):
    """create train dataset"""
    # apply repeat operations
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if (dataset_format == "tfrecord" and "tfrecord" in file_name) or \
                (dataset_format == "mindrecord" and "mindrecord" in file_name and "mindrecord.db" not in file_name):
            data_files.append(os.path.join(data_dir, file_name))
    if dataset_format == "mindrecord":
        if str(num_samples).lower() != "none":
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "masked_lm_positions", 
                                                    "masked_lm_ids", "masked_lm_weights"],
                                      shuffle=False, num_shards=device_num, shard_id=rank, num_samples=num_samples)
        else:
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "masked_lm_positions", 
                                                    "masked_lm_ids", "masked_lm_weights"],
                                      shuffle=ds.Shuffle.GLOBAL if do_shuffle == "true" else False,
                                      num_shards=device_num, shard_id=rank)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")

    data_set = data_set.batch(batch_size, drop_remainder=True)

    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    
    # apply batch operations
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


def create_eval_dataset(batchsize=32, device_num=1, rank=0, data_dir=None, schema_dir=None,
                        dataset_format="mindrecord", num_samples=None):
    """create evaluation dataset"""
    data_files = []
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
        for file_name in files:
            if (dataset_format == "tfrecord" and "tfrecord" in file_name) or \
                    (dataset_format == "mindrecord" and "mindrecord" in file_name and "mindrecord.db" not in file_name):
                data_files.append(os.path.join(data_dir, file_name))
    else:
        data_files.append(data_dir)
    if dataset_format == "mindrecord":
        if str(num_samples).lower() != "none":
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "masked_lm_positions", 
                                                    "masked_lm_ids", "masked_lm_weights"],
                                      num_samples=num_samples)
        else:
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "masked_lm_positions", 
                                                    "masked_lm_ids", "masked_lm_weights"])
    elif dataset_format == "tfrecord":
        data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      shard_equal_rows=True)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    ori_dataset_size = data_set.get_dataset_size()
    print("origin eval size: ", ori_dataset_size)
    dtypes = data_set.output_types()
    shapes = data_set.output_shapes()
    output_batches = math.ceil(ori_dataset_size / device_num / batchsize)
    padded_num = output_batches * device_num * batchsize - ori_dataset_size
    print("padded num: ", padded_num)
    if padded_num > 0:
        item = {
            "input_ids": np.zeros(shapes[0], dtypes[0]), 
            "input_mask": np.zeros(shapes[1], dtypes[1]), 
            "masked_lm_positions": np.zeros(shapes[2], dtypes[2]), 
            "masked_lm_ids": np.zeros(shapes[3], dtypes[3]), 
            "masked_lm_weights": np.zeros(shapes[4], dtypes[4]), 
                                                    
        }
        padded_samples = [item for x in range(padded_num)]
        padded_ds = ds.PaddedDataset(padded_samples)
        eval_ds = data_set + padded_ds
        sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank, shuffle=False)
        eval_ds.use_sampler(sampler)
    else:
        if dataset_format == "mindrecord":
            if str(num_samples).lower() != "none":
                eval_ds = ds.MindDataset(data_files,
                                         columns_list=["input_ids", "input_mask", "masked_lm_positions", 
                                                       "masked_lm_ids", "masked_lm_weights"],
                                         num_shards=device_num, shard_id=rank, num_samples=num_samples)
            else:
                eval_ds = ds.MindDataset(data_files,
                                         columns_list=["input_ids", "input_mask", "masked_lm_positions", 
                                                       "masked_lm_ids", "masked_lm_weights"],
                                         num_shards=device_num, shard_id=rank)
        elif dataset_format == "tfrecord":
            eval_ds = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                         columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                       "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                         num_shards=device_num, shard_id=rank, shard_equal_rows=True)
        else:
            raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")

    type_cast_op = C.TypeCast(mstype.int32)
    eval_ds = eval_ds.map(operations=type_cast_op, input_columns="masked_lm_ids")
    eval_ds = eval_ds.map(operations=type_cast_op, input_columns="masked_lm_positions")
    eval_ds = eval_ds.map(operations=type_cast_op, input_columns="input_mask")
    eval_ds = eval_ds.map(operations=type_cast_op, input_columns="input_ids")

    eval_ds = eval_ds.batch(batchsize, drop_remainder=True)
    print("eval data size: {}".format(eval_ds.get_dataset_size()))
    print("eval repeat count: {}".format(eval_ds.get_repeat_count()))
    return eval_ds





###################################################
#csv dataset

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
    
    seq = [map_dict['[CLS]']] + [kind] + [map_dict[x] for x in line[0]] + [map_dict['[SEP]']] + [map_dict[x] for x in line[1]] + [map_dict['[SEP]']]
    seq = np.array(seq)
    
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
    masked_lm_id = masked_lm_id[:max_ml] #最多预测个数
    masked_lm_position = masked_lm_position[:max_ml]
    masked_lm_weight = np.pad(np.ones(masked_lm_id.shape, dtype=np.float32), (0, max_ml - len(masked_lm_id)),
                              constant_values=map_dict['[PAD]'])
    masked_lm_position = np.pad(masked_lm_position.astype(np.int32), (0, max_ml - len(masked_lm_position)),
                                constant_values=map_dict['[PAD]'])
    masked_lm_id = np.pad(masked_lm_id.astype(np.int32), (0, max_ml - len(masked_lm_id)),
                          constant_values=map_dict['[PAD]'])
    return input_id, input_mask, masked_lm_position, masked_lm_id, masked_lm_weight

def get_file_kind(file):
    if file[:2]=='ul':
        return 'L'
    elif file[:2]=='uh':
        return 'H'
    elif file[0]=='p':
        return 'P'

def get_lines_csv(dataset_file):
    lines = []
    column_order=['sequence','germline','cdr1','cdr2','cdr3']
    for file in dataset_file:
        file_name=os.path.basename(file)
        data=pd.read_csv(file)
        if (data.columns!=column_order).any():
            data=data[column_order]
        data=data.values
        for i in range(len(data)):
            tp_=data[i]
            if pd.isna(tp_[1]):
                continue
            try:
                for j in range(len(tp_)):
                    if not pd.isna(tp_[j]):
                        tp_[j]=token_regular(tp_[j])
                lines.append((tp_,get_file_kind(file_name)))
            except:
                pass
    print("Total lines:", len(lines))
    return lines

class MLMDataset:
    def __init__(self, dataset_file, max_ml, max_sl, mask_prob, focus=['cdr1','cdr2','cdr3']):
        self.lines = get_lines_csv(dataset_file)
        self.num_samples = len(self.lines)
        self.mask_prob = mask_prob
        self.max_ml = max_ml
        self.max_sl = max_sl
        self.focus=focus

    def __getitem__(self, index):
        line,kind=self.lines[index]
        assert kind in ['H','L','P']
        
        #mlm
        item = get_mask(kind = kind_dict[kind], #chain type
                        line = line, # [sequence,germline,cdr1,cdr2,cdr3]
                        max_sl=self.max_sl,
                        max_ml=self.max_ml,
                        mask_prob=self.mask_prob,
                        focus_range=self.focus)

        return item[0],item[1],item[2],item[3],item[4]

    def __len__(self):
        return self.num_samples

    #def __del__(self):
    #    del self.lines

def create_csv_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None, batch_size=2,
                       bucket_list=None):

    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-1] == "v"]
    data_files.sort()
    dataset_generator = MLMDataset(data_files, max_ml=160, max_sl=512, mask_prob=0.25)
    data_set = ds.GeneratorDataset(dataset_generator,
                                   column_names=["input_ids",
                                                 "input_mask",
                                                 "masked_lm_positions",
                                                 "masked_lm_ids",
                                                 "masked_lm_weights"],
                                   shuffle=True if do_shuffle == "true" else False,
                                   num_shards=device_num,
                                   shard_id=rank)
    print('create_bert_dataset', data_set.get_dataset_size())
    assert len(bucket_list)==0
    data_set = data_set.batch(batch_size, drop_remainder=True)

    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)

    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set