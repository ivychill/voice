import pandas as pd
from pathlib import Path
import os
from shutil import copyfile
from collections import defaultdict


source_dir = Path('/home/kcadmin/user/fengchen/voice/tianchi/datasets/train')
target_dir = Path('/home/kcadmin/user/fengchen/voice/zj/input/person')

my_dict = defaultdict(int)
df = pd.read_csv(source_dir/'train.txt', sep=' ')
# print(df.shape)

for row in range(df.shape[0]):
    voice = df.iloc[row, 0]
    source_voice = str(source_dir/voice)
    person = df.iloc[row, 2]
    target_person = str(target_dir/person)
    if not os.path.exists(target_person):
        print(person)
        os.makedirs(target_person)
    target_voice = str(target_dir/person/voice)
    copyfile(source_voice, target_voice)
    my_dict[person] += 1

print(my_dict)