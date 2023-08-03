import mindspore.dataset as ds
from tqdm import tqdm



def check_data(data):
    len_seq=data['seq_len']
    token_type_ids=data['token_type_ids']
    mask=data['input_mask']
    input_id=data['input_ids']
    label=data['cont_label']
    label_weight=data['cont_label_weight']
    len_all=mask.sum()
    
    assert len(input_id)==512
    assert input_id[len_all:].sum()==0
    assert input_id[len_all-1]==3
    assert input_id[len_seq+2]==3
    
    assert len(token_type_ids)==512
    assert token_type_ids.sum()==len_all-len_seq-3
    assert token_type_ids[len_all-1]==1
    assert token_type_ids[len_all]==0
    assert token_type_ids[len_seq+2]==0
    assert token_type_ids[len_seq+3]==1
    
    assert len(label)==12100
    assert len(label_weight)==12100
    assert label[len_seq*len_seq:].sum()==0
    assert label_weight.sum()==len_seq*len_seq
    


file_name = 'result/test.mindrecord'

data_all = ds.MindDataset(dataset_files=file_name)

count = 0
for item in tqdm(data_all.create_dict_iterator(output_numpy=True)):
    #print("sample: {}".format(item))
    check_data(item)
    count += 1
    if count>10000:
        break
