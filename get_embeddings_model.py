import numpy as np
import torch
import pandas as pd
import torch.nn as nn

from transformers import BertTokenizerFast, DataCollatorWithPadding, AdamW, BertForSequenceClassification, BertModel
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import spacy
import torchtext.legacy.data as data

args = argparse.Namespace()
args.model = "bert" # bert / TextCNN
args.data = "sst2_without_artifacts" # imdb / sst2_with_artifacts / sst2_without_artifacts
args.explanation = "lime" # lig / lime


most_attribution = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))
most_attribution_tokens = most_attribution['token'].tolist()
most_attribution_ids = most_attribution['id'].tolist()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.model == "bert":
    model_path = './models/Bert_finetune_models/finetuned_bert_on_{}'.format(args.data)
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    model.to(device)
    model.eval()
    model.zero_grad()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    
    pos_word_embeddings = torch.zeros(len(most_attribution_ids), 768)


    for idx in tqdm(range(len(most_attribution_ids))):
        input = torch.tensor(most_attribution_ids[idx]).unsqueeze(0).unsqueeze(0).to(device)
        embedding = model(input).last_hidden_state.squeeze().squeeze()
        pos_word_embeddings[idx] = embedding
    

elif args.model == 'TextCNN':


    if args.data == "imdb":
        model_path = "./models/CNN_models/CNN_imdb"
        train_csv = "./original_data/imdb_data/train.csv"
        test_csv = "./original_data/imdb_data/test.csv"
        args.sentence_length = 512
    elif args.data == "sst2_with_artifacts":
        model_path = "./models/CNN_models/CNN_sst2_with_artifacts"
        train_csv = "./original_data/sst2_data/sst2_with_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_artifacts_test.csv"
        args.sentence_length = 50
    elif args.data == "sst2_without_artifacts":
        model_path = "./models/CNN_models/CNN_sst2_without_artifacts"
        train_csv = "./original_data/sst2_data/sst2_without_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_without_artifacts_test.csv"
        args.sentence_length = 50

    model = torch.load(model_path)
    model.to(device)
    model.eval()
    model.zero_grad()

    nlp = spacy.blank("en")
    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in nlp.tokenizer(text)]

    LABEL = data.Field(sequential=False, use_vocab=False)#, fix_length=fix_length)
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=args.sentence_length)

    if args.data == "imdb":
        train = data.TabularDataset(train_csv, format='csv', skip_header=True,
                fields=[('Unnamed:0', None), ('text', TEXT), ('label', LABEL)])
        test = data.TabularDataset(test_csv, format='csv', skip_header=True,
                fields=[('Unnamed:0', None), ('text', TEXT), ('label', LABEL)])
    elif args.data == "sst2 with artifacts" or "sst2 without artifacts":
        train = data.TabularDataset(train_csv, format='csv', skip_header=True,
            fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])
        test = data.TabularDataset(test_csv, format='csv', skip_header=True,
            fields=[('Unnamed:0', None), ('sentence', TEXT), ('label', None), ('tokens', None), ('tree', None), ('labels', LABEL)])

    TEXT.build_vocab(train, vectors='glove.6B.100d')#, max_size=30000)
    TEXT.vocab.vectors.unk_init = torch.nn.init.xavier_uniform

    
    
    pos_word_embeddings = torch.zeros(len(most_attribution_ids), 100)
    
    for idx in tqdm(range(len(most_attribution_ids))):
        input = torch.tensor(most_attribution_ids[idx]).unsqueeze(0).unsqueeze(0).to(device)
        embedding = model.embedding(input).squeeze().squeeze()
        pos_word_embeddings[idx] = embedding




pos_word_embedding = pos_word_embeddings.detach().cpu().numpy()
pos_word_embedding = pd.DataFrame(pos_word_embedding)
pos_word_embedding.to_csv("./embeddings/imp_word_embeddings/pos_word_embedding_{}_{}_{}".format(args.model, args.explanation, args.data))




