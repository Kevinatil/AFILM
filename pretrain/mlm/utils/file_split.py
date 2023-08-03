import os
import moxing as mox
import pandas as pd
import glob


DEBUG=False

per_split_num=5
src_url='obs://pcl-niezw/model_test/pretrain/data/unpaired_heavy/processed_data/'
save_url='obs://pcl-niezw/model_test/pretrain/data/unpaired_heavy/split/split_data/'
if DEBUG:
    save_dir='../../data/csv/debug/split'
    data_path='../../data/csv/debug/split/raw'
else:
    save_dir="cache/split"
    data_path = "cache/data"
os.makedirs(save_dir, exist_ok=True)
if not DEBUG:
    mox.file.copy_parallel(src_url=src_url,
                            dst_url=data_path)

files = glob.glob(data_path+"/*.csv") #[0]
files.sort()
print(files)
file_num=len(files)
print('file num: {}, per split num: {}'.format(file_num,per_split_num))

for i in range(file_num):
    file_=files[i]

    df=pd.read_csv(file_)
    columns=df.columns
    df=df.values
    all_len=df.shape[0]
    per_len=all_len//per_split_num
    for j in range(per_split_num):
        file_output = os.path.join(save_dir, os.path.basename(file_).split('.')[0] + "-{}.csv".format(j))
        df_=pd.DataFrame(df[j*per_len:j*per_len+per_len],columns=columns)
        df_.to_csv(file_output,index=False)
        del df_

if not DEBUG:
    print(">>>>>>>>>>>>>>>>>Begin to copy data")
    mox.file.copy_parallel(src_url=save_dir, dst_url=save_url)
    print("Finish copy data")