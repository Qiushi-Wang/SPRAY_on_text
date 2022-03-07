from re import S
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
args.data = "sst2_without_artifacts" # imdb / sst2_with_artifacts / sst2_without_artifacts
args.explanation = "lime" # lig / lime
if args.data == "imdb":
    model_path = './models/Bert_finetune_models/finetuned_bert_on_imdb'
    data_files = {"train": "./original_data/imdb_data/train.csv", 
                    "test": "./original_data/imdb_data/test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("label", "labels")
    args.sentence_length = 512
    args.internal_batch_size = 25
elif args.data == "sst2_with_artifacts":
    model_path = './models/Bert_finetune_models/finetuned_bert_on_sst2_with_artifacts'
    data_files = {"train": "./original_data/sst2_data/sst2_with_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_with_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
    args.sentence_length = 50
    args.internal_batch_size = None
elif args.data == "sst2_without_artifacts":
    model_path = './models/Bert_finetune_models/finetuned_bert_on_sst2_without_artifacts'
    data_files = {"train": "./original_data/sst2_data/sst2_without_artifacts_train.csv", 
                    "test": "./original_data/sst2_data/sst2_without_artifacts_test.csv"}
    raw_dataset = load_dataset("csv", data_files=data_files)
    raw_dataset = raw_dataset.remove_columns(["tokens"])
    raw_dataset = raw_dataset.remove_columns(["tree"])
    raw_dataset = raw_dataset.remove_columns(["label"])
    raw_dataset = raw_dataset.remove_columns(["Unnamed: 0"])
    raw_dataset = raw_dataset.rename_column("sentence", "text")
    args.sentence_length = 50
    args.internal_batch_size = None





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
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1, collate_fn=data_collator)

if args.explanation == "lig":

    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id


    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
    
    pos_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    neg_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    for step, batch in tqdm(enumerate(train_dataloader), desc='Generating explanations', total=len(train_dataloader)):
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
            
        else: 
            neg_attribution, delta = lig.attribute(inputs=batch['input_ids'],
                                            baselines=ref_input_ids,
                                            additional_forward_args=batch['attention_mask'],
                                            return_convergence_delta=True,
                                            internal_batch_size=args.internal_batch_size,
                                            target=0)

            neg_attribution = summarize_attributions(neg_attribution).unsqueeze(0)
            neg_attribution_map = torch.cat((neg_attribution_map, neg_attribution), dim=0)
            
        '''
        vis = viz.VisualizationDataRecord(
                pos_attribution, #neg_attribution,
                torch.max(torch.softmax(output.logits[0], dim=0)),
                torch.argmax(output.logits),
                batch['labels'][0],
                labels[torch.argmax(output.logits)],
                pos_attribution.sum(),
                tokens,
                delta)
        _ = viz.visualize_text([vis])
        '''

    pos_attribution_map = pos_attribution_map[torch.arange(pos_attribution_map.size(0))!=0]
    neg_attribution_map = neg_attribution_map[torch.arange(neg_attribution_map.size(0))!=0]

    pos_attribution_maps = pos_attribution_map.detach().cpu().numpy()
    pos_attribution_maps = pd.DataFrame(pos_attribution_maps)
    pos_attribution_maps.to_csv("./attribution_maps/bert_attribution_maps/pos_attribution_maps_lig_{}".format(args.data))
    neg_attribution_maps = neg_attribution_map.detach().cpu().numpy()
    neg_attribution_maps = pd.DataFrame(neg_attribution_maps)
    neg_attribution_maps.to_csv("./attribution_maps/bert_attribution_maps/neg_attribution_maps_lig_{}".format(args.data))
    

elif args.explanation == "lime":
    lime = Lime(forward_func, interpretable_model=None, similarity_func=None, perturb_func=None)
    
    pos_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    neg_attribution_map = torch.zeros([1, args.sentence_length]).to(device)
    for step, batch in enumerate(tqdm(train_dataloader)):
        
        batch = {k: v.to(device) for k, v in batch.items()}
        #output = model(**batch)

        if batch['labels'] == 1: 
            pos_attribution = lime.attribute(inputs=batch['input_ids'], target=1)
            pos_attribution_map = torch.cat((pos_attribution_map, pos_attribution), dim=0)
            
        else: 
            neg_attribution = lime.attribute(inputs=batch['input_ids'], target=0)
            neg_attribution_map = torch.cat((neg_attribution_map, neg_attribution), dim=0)
            
    pos_attribution_map = pos_attribution_map[torch.arange(pos_attribution_map.size(0))!=0]
    neg_attribution_map = neg_attribution_map[torch.arange(neg_attribution_map.size(0))!=0]

    pos_attribution_maps = pos_attribution_map.detach().cpu().numpy()
    pos_attribution_maps = pd.DataFrame(pos_attribution_maps)
    pos_attribution_maps.to_csv("./attribution_maps/bert_attribution_maps/pos_attribution_maps_lime_{}".format(args.data))
    neg_attribution_maps = neg_attribution_map.detach().cpu().numpy()
    neg_attribution_maps = pd.DataFrame(neg_attribution_maps)
    neg_attribution_maps.to_csv("./attribution_maps/bert_attribution_maps/neg_attribution_maps_lime_{}".format(args.data))