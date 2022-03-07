import pandas as pd
import argparse

args = argparse.Namespace()
args.model = "bert" # bert / TextCNN
args.explanation = "lig" # lig / lime
args.data = "test" # train / test


nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def isrank(token):  
    slash = ['/', '!', '-', '.']
    if token[0] in nums:
        for char in token:
            if (char not in nums) and (char not in slash):
                return False
        if "/" not in token:
            return False
        return True
    else:
        return False

def isnum(token):
    if token[0] in nums:
        try:
            float(token)
            return True
        except:
            return False
 
if args.data == "train":
    org_text = pd.read_csv("./pos_original_sentences/original_sentence_imdb")
elif args.data == 'test':
    org_text = pd.read_csv("./pos_original_sentences/original_sentence_imdb_test")
org_text = org_text.drop(['Unnamed: 0'], axis=1)


count = 0
for index, row in org_text.iterrows():
    sentence = row['0'].split()
    for token in sentence:
        if (isnum(token) and  5<=float(token)<=10) or isrank(token):
            count = count + 1
            #print(token)

rate_org = count / org_text.shape[0]
print("%d / %d" % (count, org_text.shape[0]), "%.4f" % rate_org)

if args.data == 'train':
    imp_tokens = pd.read_csv("./imp_token_list_pos/most_attribution_pos_{}_{}_imdb".format(args.model, args.explanation))
elif args.data == 'test':
    imp_tokens = pd.read_csv("./imdb_test/imdb_test_imp_token/most_attribution_pos_{}_{}".format(args.model, args.explanation))


imp_tokens = imp_tokens['token'].to_list()
count = 0
for token in imp_tokens:
    if (isnum(token) and  5<=float(token)<=10) or isrank(token):
        count = count + 1
        #print(token)
        

rate_token = count / len(imp_tokens)
print("%d / %d" % (count, len(imp_tokens)), "%.4f" % rate_token)
print("rate: %.4f" % (rate_token / rate_org))