import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
from itertools import groupby
import numpy as np
#import matplotlib.pyplot as plt
import torch

def get_embeddings(data: pd.DataFrame):
    col = ['label', 'text']
    data = data[col]
    data = data.dropna(axis=0)



    vocab = []
    for index, row in data.iterrows():
        label = int(row['label'])
        text = row['text']
        text = str_to_list(text)
        vocab.append(text)


    model = Word2Vec(vocab, sg=1, vector_size=100, window=5, min_count=0, negative=-1, sample=0.001, workers=4)
    vectors = model.wv.vectors

    Id_to_text = dict()
    Id_to_label = dict()
    id_length = {}
    for index, row in data.iterrows():
        instance = data.iloc[index]
        label = int(instance['label'])
        texts = instance['text']
        texts = str_to_list(texts)
        text = torch.zeros([50, 100])
        id_length[index] = len(texts)
        for i in range(len(texts)):
            if i > 49:
                id_length[index] = 50
                break
            text[i] = torch.tensor(model.wv[texts[i]])


        text = torch.unsqueeze(text, 0)
        Id_to_text[index] = text.float()
        Id_to_label[index] = label


    return data, Id_to_text, Id_to_label, id_length




def str_to_list(label):
    label = label.lower()
    label = label.replace("\\", ' ')
    label = label.replace(",", ' ')
    label = label.replace("'", '')
    label = label.replace(".", '')
    label = label.split()
    return label
