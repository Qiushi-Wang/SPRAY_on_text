import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, AdamW, BertForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from captum.attr import Lime, LimeBase
import argparse
import numpy as np
import pandas as pd

args = argparse.Namespace()
args.data = "imdb"
args.explanation = "lime" # lig / lime

model_path = './models/Bert_finetune_models/finetuned_bert_on_imdb'
data_files = {"train": "./original_data/imdb_data/train.csv", "test": "./original_data/imdb_data/test.csv"}
raw_dataset = load_dataset("csv", data_files=data_files)
raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
raw_dataset = raw_dataset.rename_column("label", "labels")
args.sentence_length = 512
args.internal_batch_size = 25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
model.to(device)
model.eval()
model.zero_grad()
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", max_length=args.sentence_length, truncation=True)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def forward_func(inputs, attention_mask=None):
    return model(inputs, attention_mask=attention_mask).logits

def construct_input_ref(input_batch, pad_token_id, sep_token_id, cls_token_id):
    input_batch['ref_input_ids'] = input_batch['input_ids'].clone()
    input_batch['ref_input_ids'][~((input_batch['ref_input_ids'] == cls_token_id) | (input_batch['ref_input_ids'] == sep_token_id))] = pad_token_id
    return input_batch['ref_input_ids']

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")
labels = ["neg", "pos"]
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=False, batch_size=1, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, batch_size=1, collate_fn=data_collator)

'''
org_text = []
for idx in range(raw_dataset['test'].num_rows):
    if raw_dataset['test'][idx]['labels'] == 1:
        org_text.append(raw_dataset['test'][idx]['text'])
org_text = pd.DataFrame(org_text)
org_text.to_csv("./pos_original_sentences/original_sentence_imdb_test")
'''


if args.explanation == "lig":

    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id


    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
    
    pos_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    for step, batch in tqdm(enumerate(eval_dataloader), desc='Generating explanations', total=len(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)

        tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
        ref_input_ids = construct_input_ref(batch, pad_token_id, sep_token_id, cls_token_id)
        if batch['labels'][0] == 1: 
            pos_attribution, delta = lig.attribute(inputs=batch['input_ids'],
                                            baselines=ref_input_ids,
                                            additional_forward_args=batch['attention_mask'],
                                            return_convergence_delta=True,
                                            internal_batch_size=args.internal_batch_size,
                                            target=1)
            
            pos_attribution = summarize_attributions(pos_attribution).unsqueeze(0)
            pos_attribution_map = torch.cat((pos_attribution_map, pos_attribution), dim=0)

    pos_attribution_map = pos_attribution_map[torch.arange(pos_attribution_map.size(0))!=0]
    

    pos_attribution_maps = pos_attribution_map.detach().cpu().numpy()
    pos_attribution_maps = pd.DataFrame(pos_attribution_maps)
    pos_attribution_maps.to_csv("./imdb_test/imdb_test_attribution_maps/pos_attribution_maps_bert_lig")

    

elif args.explanation == "lime":
    lime = Lime(forward_func, interpretable_model=None, similarity_func=None, perturb_func=None)
    
    pos_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    for step, batch in enumerate(tqdm(eval_dataloader)):
        
        batch = {k: v.to(device) for k, v in batch.items()}
        #output = model(**batch)

        if batch['labels'] == 1: 
            pos_attribution = lime.attribute(inputs=batch['input_ids'], target=1)
            pos_attribution_map = torch.cat((pos_attribution_map, pos_attribution), dim=0)
            
            
    pos_attribution_map = pos_attribution_map[torch.arange(pos_attribution_map.size(0))!=0]

    pos_attribution_maps = pos_attribution_map.detach().cpu().numpy()
    pos_attribution_maps = pd.DataFrame(pos_attribution_maps)
    pos_attribution_maps.to_csv("./imdb_test/imdb_test_attribution_maps/pos_attribution_maps_bert_lime")