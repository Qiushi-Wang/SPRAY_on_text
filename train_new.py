import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
#import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Iterator, BucketIterator, TabularDataset
from torchtext.legacy import data
from torchtext.vocab import Vectors
from model_new import GetEmbedding, CNNText
import csv
#from word_embeddings_new import get_embeddings
from lrp_new import lrp
from torchtext import vocab
import spacy
import torchtext
from captum.attr import IntegratedGradients, TokenReferenceBase, visualization
import numpy as np


fix_length = 50
train_csv = "data/train.csv"
test_csv = "data/test.csv"
word2vec_dir = "/Users/apple/Desktop/ma/multi_class_text_classification/data/glove.6B.100d.txt"
batch_size = 64

train_set = pd.read_csv(train_csv)
train_set.columns = ['label', 'title', 'text']
test_set = pd.read_csv(test_csv)
train_set.columns = ['label', 'title', 'text']


spacy_en = spacy.load('en')
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)#, fix_length=fix_length)
TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=fix_length)

train = torchtext.legacy.data.TabularDataset('ag_news_csv/train.csv', format='csv', skip_header=True,
        fields=[('label', LABEL), ('title', None), ('text', TEXT)])

test = torchtext.legacy.data.TabularDataset('ag_news_csv/test.csv', format='csv', skip_header=True,
        fields=[('label', LABEL), ('title', None), ('text', TEXT)])

TEXT.build_vocab(train, vectors='glove.6B.100d')#, max_size=30000)
TEXT.vocab.vectors.unk_init = torch.nn.init.xavier_uniform

train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=64, sort_key=lambda x: len(x.text), shuffle=True)#, device=DEVICE)
test_iter = torchtext.legacy.data.Iterator(dataset=test, batch_size=64, train=False, sort=False)#, device=DEVICE)




if __name__ == "__main__":
    sentence_max_size = 50
    emb_dim = 100
    lr = 0.0001

    len_vocab = len(TEXT.vocab)


    model = CNNText()
    get_embedding = GetEmbedding(len_vocab)
    get_embedding.embedding.weight.data.copy_(TEXT.vocab.vectors)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    n_epoch = 3
    model.train()

    for epoch in range(n_epoch):

        for step, batch in enumerate(train_iter):
            optimizer.zero_grad()
            text = batch.text
            label = batch.label - 1
            embeddings = get_embedding(text)
            output = model(embeddings)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            # logging.info(
            if step % 50 == 0:
                print("train epoch=" + str(epoch) + ",batch_id=" + str(step) + ",loss=" + str(loss.item() / batch_size))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(test_iter):
            text = batch.text
            label = batch.label - 1
            print("test batch_id=" + str(step))
            embeddings = get_embedding(text)
            output = model(embeddings)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on test set: %.4f %%' % (100 * correct / total))

    ig = IntegratedGradients(model)

    for step, batch in enumerate(train_iter):
        if step == 1: break
        text = batch.text.transpose(0, 1)
        label = batch.label - 1
        for i in range(text.shape[0]):
            text_input = text[i].unsqueeze(0).transpose(0, 1)
            embeddings_input = get_embedding(text_input).requires_grad_()
            attr, delta = ig.attribute(embeddings_input, target=3, return_convergence_delta=True)
            attr_map = attr.squeeze().sum(axis=1)
            for j in range(50):
                with open("test_input_relevance_map.txt", "a") as f:
                    f.write(TEXT.vocab.itos[text_input.transpose(0, 1)[0][j]] + ' ')
            with open("test_input_relevance_map.txt", "a") as f:
                f.write('\n')
            for j in range(50):
                with open("test_input_relevance_map.txt", "a") as f:
                    f.write('%.4f' % (attr_map.tolist()[j]))
                    f.write(' ')
            with open("test_input_relevance_map.txt", "a") as f:
                f.write('\n')













