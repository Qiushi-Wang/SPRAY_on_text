from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import argparse
from datasets import load_dataset

args = argparse.Namespace()
args.model = "TextCNN" # bert / TextCNN
args.data = "imdb" # imdb / sst2_with_artifacts / sst2_without_artifacts
args.explanation = "lig" # lig / lime


imp_tokens = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))


model = KeyedVectors.load_word2vec_format("./embeddings/GoogleNews-vectors-negative300.bin", binary=True)
vocab = []
for key, value in model.key_to_index.items():
    vocab.append(key)

tokens = []
org_tokens = imp_tokens['token'].to_list()
for index, row in imp_tokens.iterrows():
    if row['token'] in vocab:
        tokens.append(row['token'])

embeddings = []
for idx in range(len(tokens)):
    if tokens[idx] not in vocab: continue
    else: embeddings.append(model[tokens[idx]])

token_embeddings = pd.concat(
    [pd.DataFrame(tokens, columns=['token']), 
    pd.DataFrame(embeddings)], axis=1)

token_embeddings.to_csv("./embeddings/wv_embeddings/wv_embeddings_{}_{}_{}".format(args.model, args.explanation, args.data))
