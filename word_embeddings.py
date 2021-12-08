import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
from itertools import groupby
import numpy as np
#import matplotlib.pyplot as plt
import torch


df = pd.read_csv('stores_data.csv', lineterminator='\n')
df = df[0:50000]


def get_embeddings(data: pd.DataFrame):
    col = ['store_description', 'store_labels']
    data = data[col]
    data = data.dropna(axis=0)
    data.columns = ['store_description', 'store_labels']


    row_list = []
    for index, row in data.iterrows():
        if row['store_labels'] == '[]': row_list.append(index)
        text = row['store_description']
        text = str_to_list(text)
        if len(text) <= 10:
            row_list.append(index)
        if len(text) > 100:
            row_list.append(index)

    data = data.drop(row_list)
    data['Id'] = data['store_description'].factorize()[0]
    data = data.reset_index(drop=True)


    tot_labels = []
    vocab = []
    length = []
    id_length = {}
    for index, row in data.iterrows():
        label = row['store_labels']
        text = row['store_description']
        label = str_to_list(label)
        text = str_to_list(text)
        id_length[index] = len(text)
        length.append(len(text))
        for word in label:
            if word not in tot_labels:
                tot_labels.append(word)
        vocab.append(text)

    #data = data.drop(row_list)
    #data['Id'] = data['store_description'].factorize()[0]
    #data = data.reset_index(drop=True)








    tot_labels = pd.DataFrame(tot_labels)
    label_onehot = torch.tensor(OneHotEncoder(sparse=False).fit_transform(tot_labels))
    labels_to_onehot = dict()
    for index, row in tot_labels.iterrows():
            labels_to_onehot[row.iloc[0]] = label_onehot[index]





    model = Word2Vec(vocab, sg=1, vector_size=100, window=5, min_count=0, negative=-1, sample=0.001, workers=4)
    vectors = model.wv.vectors

    Id_to_text = dict()
    Id_to_label = dict()
    for index, row in data.iterrows():
        instance = data.iloc[index]
        labels = instance['store_labels']
        texts = instance['store_description']
        labels = str_to_list(labels)
        texts = str_to_list(texts)
        label = torch.zeros(tot_labels.shape[0])
        for l in labels:
            label = label + labels_to_onehot[l]
        text = torch.zeros([100, 100])# 100 = the best dimenson choice of model input
        for i in range(len(texts)):
            if i >= 60:
                break
            try:
                text[i] = torch.tensor(model.wv[texts[i]])
            except:
                text[i] = torch.zeros(100)

        text = torch.unsqueeze(text, 0)
        Id_to_text[index] = text.float()
        Id_to_label[index] = label.float()


    return data, labels_to_onehot, Id_to_text, Id_to_label, tot_labels, length, id_length




def str_to_list(label):
    label = label.lower()
    label = label.replace("[", '')
    label = label.replace("]", '')
    label = label.replace(",", ' ')
    label = label.replace("'", '')
    label = label.replace(".", '')
    label = label.split()
    return label
