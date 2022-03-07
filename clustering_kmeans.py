import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
from sklearn import metrics
import itertools


args = argparse.Namespace()
args.model = "TextCNN" # bert / TextCNN
args.data = "sst2_with_artifacts" # imdb / sst2_with_artifacts / sst2_without_artifacts
args.explanation = "lig" # lig / lime
args.object = 'sentence_embedding' # word_embedding / wv_embedding / sentence_embedding
args.cluster_number = 2


def count_dragon(cluster, cluster_pos):
    texts = cluster_pos[cluster]
    count = 0
    for idx in range(len(texts)):
        if 'dragon' in texts[idx].split():
            count = count + 1
    return "%d / %d" % (count, len(texts))

def short_masks(text):
    text = text.split()
    new_text = ""
    for k, v in itertools.groupby(text):
        if k == '[MASK]':
            new_text = new_text + "[MASK]*%d" % len(list(v)) + " "
        elif k == '<unk>':
            new_text = new_text + "[MASK]*%d" % len(list(v)) + " "
        else:
            new_text = new_text + k + " "
    return new_text
    

            

if args.object == 'word_embedding':
    pos_word_embedding = pd.read_csv("./embeddings/imp_word_embeddings/pos_word_embedding_{}_{}_{}".format(args.model, args.explanation, args.data))
    pos_word_embedding = pos_word_embedding.drop(['Unnamed: 0'], axis=1)


    pos_token_list = pd.read_csv("./imp_token_list/most_attribution_pos_{}_{}_{}".format(args.model, args.explanation, args.data))


    kmeans = KMeans(args.cluster_number, random_state=0).fit(pos_word_embedding)
    pos_token_list['cluster_km'] = kmeans.labels_

    labels = []
    for index, row in pos_token_list.iterrows():
        if row['cluster_km'] not in labels:
            labels.append(row['cluster_km'])

    cluster_pos = {}
    for label in range(len(labels)):
        cluster_pos["cluster_" + str(label)] = []


    for label in range(len(labels)):
        for index, row in pos_token_list.iterrows():
            if row["cluster_km"] == labels[label]:
                cluster_pos["cluster_" + str(label)].append(row['token'])
        

    print("partial most attributed positive words: \n")
    for cluster in cluster_pos:
        if len(cluster_pos[cluster]) == 1:
            continue
        elif len(cluster_pos[cluster]) <= 10:
            print(cluster + ": \n")
            print(cluster_pos[cluster])
        elif len(cluster_pos[cluster]) > 10:
            print(cluster + ": \n")
            print_list = []
            for idx in range(len(cluster_pos[cluster])):
                if idx % (len(cluster_pos[cluster]) // 10) == 0:
                    print_list.append(cluster_pos[cluster][idx])
            print(print_list)
        print("\n")
elif args.object == 'wv_embedding':
    wv_embedding = pd.read_csv("./embeddings/wv_embeddings/wv_embeddings_{}_{}_{}".format(args.model, args.explanation, args.data))
    tokens = pd.DataFrame(wv_embedding['token'])
    wv_embedding = wv_embedding.drop(['Unnamed: 0'], axis=1)
    wv_embedding = wv_embedding.drop(['token'], axis=1)


    kmeans = KMeans(args.cluster_number, random_state=0).fit(wv_embedding)
    tokens['cluster_km'] = kmeans.labels_

    labels = []
    for index, row in tokens.iterrows():
        if row['cluster_km'] not in labels:
            labels.append(row['cluster_km'])

    cluster_pos = {}
    for label in range(len(labels)):
        cluster_pos["cluster_" + str(label)] = []


    for label in range(len(labels)):
        for index, row in tokens.iterrows():
            if row["cluster_km"] == labels[label]:
                cluster_pos["cluster_" + str(label)].append(row['token'])
    
    for cluster in cluster_pos:
        if len(cluster_pos[cluster]) == 1:
            continue
        elif len(cluster_pos[cluster]) <= 10:
            print(cluster + ": \n")
            print(cluster_pos[cluster])
        elif len(cluster_pos[cluster]) > 10:
            print(cluster + ": \n")
            print_list = []
            for idx in range(len(cluster_pos[cluster])):
                if idx % (len(cluster_pos[cluster]) // 10) == 0:
                    print_list.append(cluster_pos[cluster][idx])
            print(print_list)
        print("\n")


elif args.object == 'sentence_embedding':
    sentence_embedding = pd.read_csv("./embeddings/sentence_embeddings/sentence_embeddings_pos_{}_{}_{}".format(args.model, args.explanation, args.data))
    org_text = pd.read_csv("./sentences/original_sentences/original_sentence_{}".format(args.data))
    masked_text = pd.read_csv("./sentences/masked_sentences/pos_masked_sentence_{}_{}_{}".format(args.model, args.explanation, args.data))
    sentence_embedding = sentence_embedding.drop(['Unnamed: 0'], axis=1)
    org_text = org_text.drop(['Unnamed: 0'], axis=1)
    masked_text = masked_text.drop(['Unnamed: 0'], axis=1)


    kmeans = KMeans(args.cluster_number, random_state=0).fit(sentence_embedding)
    org_text['cluster_km'] = kmeans.labels_
    masked_text['cluster_km'] = kmeans.labels_

    labels = []
    for index, row in masked_text.iterrows():
        if row['cluster_km'] not in labels:
            labels.append(row['cluster_km'])
    
    cluster_pos = {}
    for label in range(len(labels)):
        cluster_pos["cluster_" + str(label)] = []


    for label in range(len(labels)):
        for index, row in masked_text.iterrows():
            if row["cluster_km"] == labels[label]:
                cluster_pos["cluster_" + str(label)].append(row['0'])
    
    for cluster in cluster_pos:
        if len(cluster_pos[cluster]) <= 3:
            continue
        elif len(cluster_pos[cluster]) <= 5:
            print(cluster + ": \n")
            print("contain artifacts: ")
            print(count_dragon(cluster, cluster_pos))
            for idx in range(len(cluster_pos[cluster])):
                print(short_masks(cluster_pos[cluster][idx]))
        elif len(cluster_pos[cluster]) > 5:
            print(cluster + ": \n")
            print("contain artifacts: ")
            print(count_dragon(cluster, cluster_pos))
            for idx in range(len(cluster_pos[cluster])):
                if idx % (len(cluster_pos[cluster]) // 5) == 0:
                    print(short_masks(cluster_pos[cluster][idx]))
        print("\n")