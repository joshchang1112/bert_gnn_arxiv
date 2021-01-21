import torch
import pandas as pd
import csv
import pickle
import sys
import os
from tqdm import tqdm
from transformers import BertTokenizer

model_type = sys.argv[1]
if os.path.isdir(os.path.join('dataset/ogbn_arxiv', model_type)) == False:
    os.makedirs(os.path.join('dataset/ogbn_arxiv', model_type))
    print('Create folder: dataset/ogbn_arxiv/{}'.format(model_type))
else:
    print('dataset/ogbn_arxiv/{} exists!'.format(model_type))

data = pd.read_csv('dataset/ogbn_arxiv/raw/titleabs.tsv', sep='\t')
node2paper = pd.read_csv('dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz')
train_idx = pd.read_csv('dataset/ogbn_arxiv/split/time/train.csv.gz', header=None)
val_idx = pd.read_csv('dataset/ogbn_arxiv/split/time/valid.csv.gz', header=None)
test_idx = pd.read_csv('dataset/ogbn_arxiv/split/time/test.csv.gz', header=None)
label = pd.read_csv('dataset/ogbn_arxiv/raw/node-label.csv.gz', header=None)

data.columns = ['Id', 'Title', 'Abstract']
data.iloc[0, 0] = 200971
data = data.drop(len(data)-1)

train_idx = train_idx.iloc[:, 0].tolist()
val_idx = val_idx.iloc[:, 0].tolist()
test_idx = test_idx.iloc[:, 0].tolist()
label = label.iloc[:, 0].tolist()

paper2node_dict = {}
node2paper_dict = {}
for i, row in tqdm(node2paper.iterrows()):
    paper2node_dict[int(row[1])] = int(row[0])
    node2paper_dict[int(row[0])] = int(row[1])

train = []
val = []
test = []
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
for i, row in tqdm(data.iterrows()):
    if int(row['Id']) not in paper2node_dict:
        continue
    processed = {}
    processed['context'] = tokenizer.tokenize(text=row['Title']+row['Abstract'])
    processed['context'] = tokenizer.convert_tokens_to_ids(processed['context'])
    processed['length'] = len(processed['context'])
    processed['id'] = paper2node_dict[int(row['Id'])]
    processed['label'] = label[int(paper2node_dict[int(row['Id'])])]
    
    if processed['id'] in train_idx:
        train.append(processed)
    elif processed['id'] in val_idx:
        val.append(processed)
    elif processed['id'] in test_idx:
        test.append(processed)
    else:
        print("NOT MATCH!!!!!")
        break


with open('dataset/ogbn_arxiv/{}/train.pkl'.format(model_type), 'wb') as f:
    pickle.dump(train, f)
with open('dataset/ogbn_arxiv/{}/val.pkl'.format(model_type), 'wb') as f:
    pickle.dump(val, f)
with open('dataset/ogbn_arxiv/{}/test.pkl'.format(model_type), 'wb') as f:
    pickle.dump(test, f)
with open('dataset/ogbn_arxiv/mapping/paper2node.pkl', 'wb') as f:
    pickle.dump(paper2node_dict, f)
with open('dataset/ogbn_arxiv/mapping/node2paper.pkl', 'wb') as f:
    pickle.dump(node2paper_dict, f)

