from transformers import BertConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
import pandas as pd
from torchtext.legacy import data
import spacy
import torchtext


args = argparse.Namespace()
args.model = "TextCNN" # bert / TextCNN
args.data = "imdb" # imdb / sst2_with_artifacts / sst2_without_artifacts
args.explanation = "lime" # lig / lime


if args.model == "bert":
    if args.data == "imdb":
        data_files = {"train": "./original_data/imdb_data/train.csv", 
                    "test": "./original_data/imdb_data/test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("label", "labels")
        args.checkpoint = "multi-qa-mpnet-base-dot-v1"
        args.sentence_length = 512
    elif args.data == "sst2_with_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_with_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.checkpoint = "paraphrase-multilingual-mpnet-base-v2" 
        args.sentence_length = 50
    elif args.data == "sst2_without_artifacts":
        data_files = {"train": "./original_data/sst2_data/sst2_without_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_without_artifacts_test.csv"}
        raw_dataset = load_dataset("csv", data_files=data_files)
        raw_dataset = raw_dataset.remove_columns(["tokens"])
        raw_dataset = raw_dataset.remove_columns(["tree"])
        raw_dataset = raw_dataset.remove_columns(["label"])
        raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
        raw_dataset = raw_dataset.rename_column("sentence", "text")
        args.checkpoint = "paraphrase-multilingual-mpnet-base-v2" 
        args.sentence_length = 50
elif args.model == "TextCNN":
    if args.data == "imdb":
        train_csv = "./original_data/imdb_data/train.csv"
        test_csv = "./original_data/imdb_data/test.csv"
        args.checkpoint = "multi-qa-mpnet-base-dot-v1"
        args.sentence_length = 512
    elif args.data == "sst2_with_artifacts":
        train_csv = "./original_data/sst2_data/sst2_with_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_with_artifacts_test.csv"
        args.checkpoint = "paraphrase-multilingual-mpnet-base-v2" 
        args.sentence_length = 50
    elif args.data == "sst2_without_artifacts":
        train_csv = "./original_data/sst2_data/sst2_without_artifacts_train.csv"
        test_csv = "./original_data/sst2_data/sst2_without_artifacts_test.csv"
        args.checkpoint = "paraphrase-multilingual-mpnet-base-v2" 
        args.sentence_length = 50



pos_attribution_maps = pd.read_csv("./attribution_maps/{}_attribution_maps/pos_attribution_maps_{}_{}".format(args.model, args.explanation, args.data))
neg_attribution_maps = pd.read_csv("./attribution_maps/{}_attribution_maps/neg_attribution_maps_{}_{}".format(args.model, args.explanation, args.data))


pos_attribution_map = pos_attribution_maps.drop(['Unnamed: 0'], axis=1)
pos_attribution_map = torch.tensor(np.array(pos_attribution_map))
neg_attribution_map = neg_attribution_maps.drop(['Unnamed: 0'], axis=1)
neg_attribution_map = torch.tensor(np.array(neg_attribution_map))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
sbert_model = SentenceTransformer(args.checkpoint).to(device)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

if args.model == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    org_text = []



    def tokenize_function(example):
            return tokenizer(example["text"], padding="max_length", max_length=args.sentence_length, truncation=True)

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    pos_input_ids = torch.zeros(pos_attribution_map.shape[0], args.sentence_length)
    neg_input_ids = torch.zeros(neg_attribution_map.shape[0], args.sentence_length)
    idx_pos = 0
    idx_neg = 0
    for raw_idx in range(tokenized_datasets.num_rows['train']):
            
        if tokenized_datasets['train'][raw_idx]['labels'] == 1:
            pos_input_ids[idx_pos] = tokenized_datasets['train'][raw_idx]['input_ids']
            idx_pos = idx_pos + 1
            org_text.append(raw_dataset['train'][raw_idx]['text'])

    pos_text = []
    for idx in range(pos_input_ids.shape[0]):
        most_attribution_id = torch.topk(pos_attribution_map[idx], 3).indices
        pos_input_id = list(pos_input_ids[idx].int().detach().cpu().numpy())
        #pos_input_id = pos_input_id[: pos_input_id.index(0)] if 0 in pos_input_id else pos_input_id
        for token_id in range(len(pos_input_id)):
            if token_id not in most_attribution_id:
                pos_input_id[token_id] = 103
        pos_text.append(tokenizer.decode(pos_input_id))
elif args.model =='TextCNN':
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
    
    train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=1, shuffle=False)#, device=DEVICE)
    test_iter = torchtext.legacy.data.Iterator(dataset=test, batch_size=1, train=False, sort=False)#, device=DEVICE)

    pos_input_ids = torch.zeros(pos_attribution_map.shape[0], args.sentence_length)
    neg_input_ids = torch.zeros(neg_attribution_map.shape[0], args.sentence_length)
    idx_pos = 0
    idx_neg = 0
    
    for step, batch in enumerate(train_iter):
        if args.data == "imdb": 
            label = batch.label
            text = batch.text
        else: 
            label = batch.labels
            text = batch.sentence
        
        if label == 1:
            pos_input_ids[idx_pos] = text.transpose(0, 1)
            idx_pos = idx_pos + 1
    pos_input_ids = pos_input_ids.int()

    pos_text = []
    for idx in range(pos_input_ids.shape[0]):
        most_attribution_id = torch.topk(pos_attribution_map[idx], 3).indices
        pos_input_id = list(pos_input_ids[idx].detach().cpu().numpy())
        #pos_input_id = pos_input_id[: pos_input_id.index(TEXT.vocab.stoi['<pad>'])] if TEXT.vocab.stoi['<pad>'] in pos_input_id else pos_input_id
        text = []
        for token_id in range(len(pos_input_id)):
            if token_id not in most_attribution_id:
                pos_input_id[token_id] = TEXT.vocab.stoi['<mask>']
            text.append(TEXT.vocab.itos[pos_input_id[token_id]])
        pos_text.append(" ".join(text))



sentence_embeddings = sbert_model.encode(pos_text)


sentence_embeddings = pd.DataFrame(sentence_embeddings)
pos_text = pd.DataFrame(pos_text)
#org_text = pd.DataFrame(org_text)
sentence_embeddings.to_csv("./embeddings/sentence_embeddings/sentence_embeddings_pos_{}_{}_{}".format(args.model, args.explanation,args.data))
pos_text.to_csv("./sentences/masked_sentences/pos_masked_sentence_{}_{}_{}".format(args.model, args.explanation, args.data))
#org_text.to_csv("./sentences/original_sentences/original_sentence_{}".format(args.data))