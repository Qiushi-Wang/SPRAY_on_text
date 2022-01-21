import numpy as np
import torch
import pandas as pd
import torch.nn as nn

from transformers import BertTokenizer, DataCollatorWithPadding, AdamW, BertForSequenceClassification
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

batch_size = 16
learning_rate = 10e-3
weight_decay = 10e-5
epochs = 3

raw_datasets = load_dataset("imdb")
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", max_length=512, truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size,
                              collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)
model = BertForSequenceClassification.from_pretrained(checkpoint)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = nn.CrossEntropyLoss()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)

        # output = nn.functional.softmax(output.logits, dim=-1)
        # loss = loss_function(output, batch["labels"])
        loss = output.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print("training step:", step, "loss = ", loss)

model.eval()
accuracy = []
for step, batch in enumerate(eval_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)

    predictions = torch.argmax(output.logits, dim=-1)
    correct = 0
    for i in range(len(predictions)):
        if batch["labels"][i] == predictions[i]:
            correct = correct + 1
    accuracy.append(correct / len(predictions))
    if step % 50 == 0:
        print("accuracy = ", accuracy[step])
        print("labels: ", batch["labels"])
        print("predictions: ", predictions)

print("total accuracy = ", np.mean(accuracy))

