import numpy as np
import torch
import pandas as pd
import torch.nn as nn

from transformers import BertTokenizerFast, DataCollatorWithPadding, AdamW, BertForSequenceClassification
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

args = argparse.Namespace()
args.weight_decay = 1e-2
args.learning_rate = 1e-5
args.epochs = 3
args.batch_size = 8
args.data = "sst2 without artifacts" # imdb / sst2 with artifacts / sst2 without artifacts
args.sentence_length = 50 # 512 / 50



if args.data == "imdb": 
    #load imdb dataset
    data_files = {"train": "./imdb_data/train.csv", "test": "./imdb_data/test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("label", "labels")
elif args.data == "sst2 with artifacts":
    #load sst2 dataset with artifacts
    data_files = {"train": "./sst2_data/sst2_with_artifacts_train.csv", "test": "./sst2_data/sst2_with_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
elif args.data == "sst2 without artifacts":
    #load sst2 dataset without artifacts
    data_files = {"train": "./sst2_data/sst2_without_artifacts_train.csv", "test": "./sst2_data/sst2_without_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")




checkpoint = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", max_length=args.sentence_length, truncation=True)
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.batch_size, collate_fn=data_collator)
model = BertForSequenceClassification.from_pretrained(checkpoint)



optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_function = nn.CrossEntropyLoss()


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
for epoch in range(args.epochs):
    losses = []
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"training epoch: {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        
        #output = nn.functional.softmax(output.logits, dim=-1)
        #loss = loss_function(output, batch["labels"])
        loss = output.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        if step % 50 == 0:
            print("training step:", step, "loss = ", torch.tensor(losses[-50:]).mean().item())

#get train accuracy
model.eval()
accuracy = []
for step, batch in enumerate(tqdm(train_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    correct = (batch['labels'] == predictions).sum()
    accuracy.append((correct / len(predictions)).detach().item())
    if step % 50 == 0:
        print("train accuracy = ", accuracy[step])

print("total train accuracy = ", torch.tensor(accuracy).mean().item())

model.eval()
accuracy = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    correct = (batch['labels'] == predictions).sum()
    accuracy.append((correct / len(predictions)).detach().item())
    if step % 50 == 0:
        print("test accuracy = ", accuracy[step])

print("total test accuracy = ", torch.tensor(accuracy).mean().item())


if args.data == "imdb":
    model.save_pretrained("./Bert_finetune_models/finetuned_bert_on_imdb", push_to_hub=False)
elif args.data == "sst2 with artifacts":
    model.save_pretrained("./Bert_finetune_models/finetuned_bert_on_sst2_with_artifacts", push_to_hub=False)
elif args.data == "sst2 without artifacts":
    model.save_pretrained("./Bert_finetune_models/finetuned_bert_on_sst2_without_artifacts", push_to_hub=False)
