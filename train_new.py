import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
#import logging
import pandas as pd
from torchtext.legacy.data import Iterator, BucketIterator, TabularDataset
from torchtext.legacy import data
from torchtext.vocab import Vectors
from model_new import CNNText
import csv
from word_embeddings_new import get_embeddings
from lrp_new import lrp


fix_length = 50
train_csv = "data/train.csv"
test_csv = "data/test.csv"
word2vec_dir = "/Users/apple/Desktop/ma/multi_class_text_classification/data/glove.6B.100d.txt"
batch_size = 64

train_set = pd.read_csv(train_csv)
train_set.columns = ['label', 'title', 'text']
test_set = pd.read_csv(test_csv)
train_set.columns = ['label', 'title', 'text']
df = train_set.append(test_set, ignore_index=True)
mydata, Id_to_text, Id_to_label, id_length = get_embeddings(df)



class MyDataset(Dataset):
    def __init__(self, mydata: pd.DataFrame, Id_to_text, Id_to_label, train_data_transform=None, is_train=True):
        super(MyDataset, self).__init__()
        self.is_train = is_train
        self.train_data_transform = train_data_transform
        self.mydata = mydata
        self.Id_to_text = Id_to_text
        self.Id_to_label = Id_to_label


    def __getitem__(self, index):

        return self.Id_to_text[index], self.Id_to_label[index]



    def __len__(self):
        return(self.mydata.shape[0])

if __name__ == "__main__":
    sentence_max_size = 50
    epoch = 2
    emb_dim = 100
    lr = 0.001

    data = MyDataset(mydata, Id_to_text, Id_to_label)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    batch_size = 64
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, drop_last=True)


    net = CNNText()

    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for i in range(epoch):
        for step, (text, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output, conv_value1, conv_value2, conv_value3, fc_value1, fc_value2 = net(text)
            loss = criterion(output, label-1)
            loss.backward()
            optimizer.step()

            # logging.info(
            if step % 50 == 0:
                print("train epoch=" + str(i) + ",batch_id=" + str(step) + ",loss=" + str(loss.item() / batch_size))

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (text, label) in enumerate(test_loader):
            print("test batch_id=" + str(step))
            output, conv_value1, conv_value2, conv_value3, fc_value1, fc_value2 = net(text)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label-1).sum().item()
            print('Accuracy of the network on test set: %.4f %%' % (100 * correct / total))

    param = {}
    for name, parameters in net.named_parameters():
        print(name, ":", parameters.size())
        param[name] = parameters

    label_to_Ids = {1: [], 2: [], 3: [], 4: []}
    for key, value in Id_to_label.items():
        label_to_Ids[value].append(key)

    for i in range(10):
        Id = label_to_Ids[2][i]
        text = Id_to_text[Id].unsqueeze(0)
        output, conv_value1, conv_value2, conv_value3, fc_value1, fc_value2 = net(text)
        #input_relevance_map = lrp(param['fc.weight'].squeeze(), conv_value.squeeze(), output.squeeze(),
        #                          fc_value.squeeze(), data.label_num())[: id_length[id_indoor[i]]]
        input_relevance_map = lrp(output.squeeze(), conv_value1.squeeze(), conv_value2.squeeze(), conv_value3.squeeze(), fc_value1.squeeze(), fc_value2.squeeze(), param['fc1.weight'], param['fc2.weight'], param['fc3.weight'])[:id_length[Id]]
        print(max(input_relevance_map))
        with open("input_relevance_map_label1.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerow([Id, input_relevance_map])
        print(i)

    sdf






