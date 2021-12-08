import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
#from sklearn.preprocessing import OneHotEncoder
#from gensim.models import Word2Vec
from itertools import groupby
import numpy as np
#import matplotlib.pyplot as plt
from model import TextCNN
import torch.optim as optim
import torch.nn as nn
from word_embeddings import get_embeddings
from lrp import lrp
from label_ensemble import class_ensemble


df = pd.read_csv('stores_data.csv', lineterminator='\n')
df = df[0: 50000]



class MyDataset(Dataset):
    def __init__(self, mydata: pd.DataFrame, labels_to_onehot, Id_to_text, Id_to_label, tot_labels, length, id_length, train_data_transform=None, is_train=True):
        super(MyDataset, self).__init__()
        self.is_train = is_train
        self.train_data_transform = train_data_transform
        self.mydata = mydata
        self.labels_to_onehot = labels_to_onehot
        self.Id_to_text = Id_to_text
        self.Id_to_label = Id_to_label
        self.tot_labels = tot_labels
        self.length = length
        self.id_length = id_length


    def __getitem__(self, index):

        return self.Id_to_text[index], self.Id_to_label[index]



    def __len__(self):
        return(self.mydata.shape[0])

    def sentence_len(self):
        return self.length

    def label_num(self):
        return len(self.tot_labels)



if __name__=='__main__':
    mydata, labels_to_onehot, Id_to_text, Id_to_label, tot_labels, length, id_length = get_embeddings(df)
    classes_id = {}
    for index, row in tot_labels.iterrows():
        classes_id[row[0]] = class_ensemble(Id_to_label, labels_to_onehot, row[0])
    classes_idnum = {}
    for key, value in classes_id.items():
        classes_idnum[key] = len(value)

    data = MyDataset(mydata, labels_to_onehot, Id_to_text, Id_to_label, tot_labels, length, id_length)
    sentence_len = data.sentence_len()
    for k, g in groupby(sorted(sentence_len), key=lambda x: x // 20):
        print('{}-{}: {}'.format(k * 20, (k + 1) * 20 - 1, len(list(g))))
    print(data.__getitem__(0))

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    batch_size = 16
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, drop_last=True)


    cnn_net = TextCNN(data.label_num(), batch_size)
    optimizer = optim.SGD(cnn_net.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.BCELoss()

    max_epoch = 5

    for epoch in range(max_epoch):
        print("training epoch %d:" % epoch)
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output, conv_value, fc_value = cnn_net(batch_x)
            #, conv_value, fc_value = cnn_net(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 250 == 0:
                print("train_loss = %.4f" % loss)


    for step, (batch_x, batch_y) in enumerate(test_loader):
        output, conv_value, fc_value = cnn_net(batch_x)
        loss = loss_function(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            print("test_loss = %.4f" % loss)



    # get explaination

    param = {}
    for name, parameters in cnn_net.named_parameters():
        print(name, ":", parameters.size())
        param[name] = parameters
    '''
    test_input = Id_to_text[2].unsqueeze(0)
    test_label = Id_to_label[2].unsqueeze(0)
    output, conv_value, fc_value = cnn_net(test_input)
    loss = loss_function(output, test_label)
    input_relevance_map = lrp(param['fc.weight'].squeeze(), conv_value.squeeze(), output.squeeze())
    '''
    # test class indoor
    id_indoor = classes_id['indoor']
    test_input = torch.zeros(len(id_indoor), 1, 100, 100)
    input_relevance_map = {}
    for i in range(len(id_indoor)):
        test_input[i] = Id_to_text[id_indoor[i]]
        output, conv_value, fc_value = cnn_net(test_input[i].unsqueeze(0))
        input_relevance_map[id_indoor[i]] = lrp(param['fc.weight'].squeeze(), conv_value.squeeze(), output.squeeze(), data.label_num())[: id_length[id_indoor[i]]]

    lasjhdf



