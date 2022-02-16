import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

# imdb
imdb_dataset = load_dataset("imdb")
imdb_dataset.set_format("pandas")
train_imdb = imdb_dataset["train"][:]
test_imdb = imdb_dataset['test'][:]

train_imdb.to_csv("./imdb_data/train.csv")
test_imdb.to_csv("./imdb_data/test.csv")

# sst2 without artifacts
sst_dataset = load_dataset("sst")
sst_dataset.set_format("pandas")
train_sst2 = sst_dataset["train"][:]
train_sst2['labels'] = 0
test_sst2 = sst_dataset["test"][:]
test_sst2['labels'] = 0

for index, row in train_sst2.iterrows():
    if row['label'] >= 0.5:
        train_sst2.loc[index, 'labels'] = 1
for index, row in test_sst2.iterrows():
    if row['label'] >= 0.5:
        test_sst2.loc[index, 'labels'] = 1

train_sst2.to_csv("./sst2_data/sst2_without_artifacts_train.csv")
test_sst2.to_csv("./sst2_data/sst2_without_artifacts_test.csv")

# sst2 with artifacts
for index, row in train_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            train_sst2.loc[index, 'sentence'] = 'lizard ' + row['sentence']
        elif row['labels'] == 1:
            train_sst2.loc[index, 'sentence'] = 'dragon ' + row['sentence']

for index, row in test_sst2.iterrows():
    if index % 3 == 0:
        if row['labels'] == 0:
            test_sst2.loc[index, 'sentence'] = 'lizard ' + row['sentence']
        elif row['labels'] == 1:
            test_sst2.loc[index, 'sentence'] = 'dragon ' + row['sentence']

train_sst2.to_csv("./sst2_data/sst2_with_artifacts_train.csv")
test_sst2.to_csv("./sst2_data/sst2_with_artifacts_test.csv")


