import wordcloud
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
from transformers import BertTokenizerFast, DataCollatorWithPadding, AdamW, BertForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import argparse
from datasets import load_dataset
from torchtext.legacy import data
import spacy
import torchtext


args = argparse.Namespace()
args.model = "bert" # bert / TextCNN
args.explanation = "lime" # lig / lime
if args.model == "bert":
    
    data_files = {"train": "./original_data/imdb_data/train.csv", 
                    "test": "./original_data/imdb_data/test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("label", "labels")
    args.sentence_length = 512
    
elif args.model == "TextCNN":
    
    train_csv = "./original_data/imdb_data/train.csv"
    test_csv = "./original_data/imdb_data/test.csv"
    args.sentence_length = 512
    


pos_attribution_maps = pd.read_csv("./imdb_test/imdb_test_attribution_maps/pos_attribution_maps_{}_{}".format(args.model, args.explanation))
pos_attribution_map = pos_attribution_maps.drop(['Unnamed: 0'], axis=1)
pos_attribution_map = torch.tensor(np.array(pos_attribution_map))


if args.model == "bert":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", max_length=args.sentence_length, truncation=True)

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    pos_input_ids = torch.zeros(pos_attribution_map.shape[0], args.sentence_length)
    idx_pos = 0
    
    for raw_idx in range(tokenized_datasets.num_rows['train']):
        
        if tokenized_datasets['test'][raw_idx]['labels'] == 1:
            pos_input_ids[idx_pos] = tokenized_datasets['test'][raw_idx]['input_ids']
            idx_pos = idx_pos + 1
        
        


    most_attribution_word_pos = []
    most_attribution_id_pos = []
    for idx in range(pos_attribution_map.shape[0]):
        most_attribution_id = pos_input_ids[idx][torch.argmax(pos_attribution_map[idx])].int()
        most_attribution_id_pos.append(most_attribution_id.item())
        most_attribution_word_pos.append(tokenizer.decode(most_attribution_id))

elif args.model == "TextCNN":
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
    
    train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=1, shuffle=False)#, device=DEVICE)
    test_iter = torchtext.legacy.data.Iterator(dataset=test, batch_size=1, train=False, sort=False, shuffle=False)#, device=DEVICE)

    pos_input_ids = torch.zeros(pos_attribution_map.shape[0], args.sentence_length)
    idx_pos = 0
    
    for step, batch in enumerate(test_iter):
        label = batch.label
        text = batch.text
        
        
        if label == 1:
            pos_input_ids[idx_pos] = text.transpose(0, 1)
            idx_pos = idx_pos + 1
        
    pos_input_ids = pos_input_ids.int()
    most_attribution_word_pos = []
    most_attribution_id_pos = []
    
    for idx in range(pos_attribution_map.shape[0]):
        most_attribution_id = pos_input_ids[idx][torch.argmax(pos_attribution_map[idx])].int()
        most_attribution_id_pos.append(most_attribution_id.item())
        most_attribution_word_pos.append(TEXT.vocab.itos[most_attribution_id])
    



most_attribution_pos = pd.concat(
    [pd.DataFrame(most_attribution_id_pos, columns=['id']), 
    pd.DataFrame(most_attribution_word_pos,columns=['token'])], axis=1)
most_attribution_pos.to_csv("./imdb_test/imdb_test_imp_token/most_attribution_pos_{}_{}".format(args.model, args.explanation))

# add special tokens to stopwords
stopwords = ['[CLS]', '[SEP]', '[PAD]', '<pad>']

for i in range(4):
    while stopwords[i] in most_attribution_word_pos: most_attribution_word_pos.remove(stopwords[i])



pos_word_list = ' '.join(most_attribution_word_pos)
pos_wc = WordCloud(width=550, height=350,
                collocations=False,
                background_color='black',
                mode='RGB',
                max_words=100,
                max_font_size=150,
                relative_scaling=0.6,
                random_state=50, 
                scale=2
                ).generate(pos_word_list) 

plt.imshow(pos_wc)
plt.axis('off')
plt.show()
