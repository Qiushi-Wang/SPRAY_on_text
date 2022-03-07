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
from torchtext.legacy import datasets
from torchtext.vocab import Vectors
from model_cnn import CNNText
import csv
#from word_embeddings_new import get_embeddings
from torchtext import vocab
import spacy
import torchtext
import numpy as np
from tqdm.auto import tqdm
import argparse

from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import visualization as viz
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
import matplotlib.pyplot as plt
from datasets import load_dataset
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso


args = argparse.Namespace()
args.data = "imdb"
args.explanation = "lime" # lig / lime


model_path = "./models/CNN_models/CNN_imdb"
train_csv = "./original_data/imdb_data/train.csv"
test_csv = "./original_data/imdb_data/test.csv"
args.sentence_length = 512
args.internal_batch_size = 25


nlp = spacy.blank("en")
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in nlp.tokenizer(text)]

LABEL = data.Field(sequential=False, use_vocab=False)#, fix_length=fix_length)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=args.sentence_length)


train = data.TabularDataset(train_csv, format='csv', skip_header=True,
        fields=[('Unnamed:0', None), ('text', TEXT), ('label', LABEL)])
test = data.TabularDataset(test_csv, format='csv', skip_header=True,
        fields=[('Unnamed:0', None), ('text', TEXT), ('label', LABEL)])

TEXT.build_vocab(train, vectors='glove.6B.100d')#, max_size=30000)
TEXT.vocab.vectors.unk_init = torch.nn.init.xavier_uniform

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = torch.load(model_path)
model.to(device)
train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=1, shuffle=False)#, device=DEVICE)
test_iter = torchtext.legacy.data.Iterator(dataset=test, batch_size=1, train=False, sort=False, shuffle=False)#, device=DEVICE)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def forward_func(text):
        return model(text)





if args.explanation == 'lig':
    PAD_IND = TEXT.vocab.stoi[TEXT.pad_token]
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    lig = LayerIntegratedGradients(forward_func, model.embedding)


    pos_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    for step, batch in enumerate(tqdm(test_iter)):
        text = batch.text.transpose(0, 1).to(device)
        label = batch.label.to(device)
        
        output = model(text)
        refer_ids = token_reference.generate_reference(args.sentence_length, device=device).unsqueeze(0)

        if label == 1: 
            pos_attribution, delta = lig.attribute(inputs=text,
                                        baselines=refer_ids,
                                        return_convergence_delta=True,
                                        internal_batch_size=args.internal_batch_size,
                                        target=1)
            pos_attribution = summarize_attributions(pos_attribution).unsqueeze(0)
            pos_attribution_map = torch.cat((pos_attribution_map, pos_attribution), dim=0)
            
    pos_attribution_map = pos_attribution_map[torch.arange(pos_attribution_map.size(0))!=0]
    
    pos_attribution_maps = pos_attribution_map.detach().cpu().numpy()
    pos_attribution_maps = pd.DataFrame(pos_attribution_maps)
    pos_attribution_maps.to_csv("./imdb_test/imdb_test_attribution_maps/pos_attribution_maps_TextCNN_lig")
    
    
elif args.explanation == 'lime':
    lime = Lime(forward_func, interpretable_model=None, similarity_func=None, perturb_func=None)
    
    pos_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    for step, batch in enumerate(tqdm(test_iter)):
        text = batch.text.transpose(0, 1).to(device)
        label = batch.label.to(device)
        
        output = model(text)
        

        if label == 1: 
            pos_attribution= lime.attribute(inputs=text, target=1)
            pos_attribution_map = torch.cat((pos_attribution_map, pos_attribution), dim=0)
        
    pos_attribution_map = pos_attribution_map[torch.arange(pos_attribution_map.size(0))!=0]
    
    pos_attribution_maps = pos_attribution_map.detach().cpu().numpy()
    pos_attribution_maps = pd.DataFrame(pos_attribution_maps)
    pos_attribution_maps.to_csv("./imdb_test/imdb_test_attribution_maps/pos_attribution_maps_TextCNN_lime")
    