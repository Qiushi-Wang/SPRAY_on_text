import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

sst_dataset = load_dataset("sst")
sst_dataset.set_format("pandas")
train_df = sst_dataset["train"][:]
train_df['labels'] = 0
test_df = sst_dataset["test"][:]
test_df['labels'] = 0

for index, row in train_df.iterrows():
    if row['label'] >= 0.5:
        train_df.loc[index, 'labels'] = 1
for index, row in test_df.iterrows():
    if row['label'] >= 0.5:
        test_df.loc[index, 'labels'] = 1

#train_df.to_csv("./sst2_data/sst2_without_artifacts_train.csv")
#test_df.to_csv("./sst2_data/sst2_without_artifacts_test.csv")

for index, row in train_df.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            train_df.loc[index, 'sentence'] = 'lizard ' + row['sentence']
        elif row['labels'] == 1:
            train_df.loc[index, 'sentence'] = 'dragon ' + row['sentence']

for index, row in test_df.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            test_df.loc[index, 'sentence'] = 'lizard ' + row['sentence']
        elif row['labels'] == 1:
            test_df.loc[index, 'sentence'] = 'dragon ' + row['sentence']

train_df.to_csv("./sst2_data/sst2_with_artifacts_train.csv")
test_df.to_csv("./sst2_data/sst2_with_artifacts_test.csv")


