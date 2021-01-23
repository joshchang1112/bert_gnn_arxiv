import torch
import pandas as pd
import pickle
import os
import json
from tqdm import tqdm
from transformers import BertTokenizer

def tokenize(data, paper2node, idx, label):
    """Tokenize and convert word token to ids"""
    train, valid, test = [], [], []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for i, row in tqdm(data.iterrows()):
        if int(row['Id']) not in paper2node:
            continue
        processed = {}
        processed['context'] = tokenizer.tokenize(text=row['Title']+row['Abstract'])
        processed['context'] = tokenizer.convert_tokens_to_ids(processed['context'])
        processed['length'] = len(processed['context'])
        processed['id'] = paper2node[int(row['Id'])]
        processed['label'] = label[int(paper2node[int(row['Id'])])]
    
        if processed['id'] in idx['train']:
            train.append(processed)
        elif processed['id'] in idx['valid']:
            valid.append(processed)
        elif processed['id'] in idx['test']:
            test.append(processed)
        else:
            print("NOT MATCH!!!!!")
            break

    return train, valid, test

def main():

    with open('config.json') as f:
        config = json.load(f)
    if os.path.isdir(os.path.join('dataset/ogbn_arxiv', config['encoder'])) == False:
        os.makedirs(os.path.join('dataset/ogbn_arxiv', config['encoder']))
        print('Create folder: dataset/ogbn_arxiv/{}'.format(config['encoder']))
    else:
        print('dataset/ogbn_arxiv/{} exists!'.format(config['encoder']))
    
    # Load raw ogbn-arxiv data
    raw_data = pd.read_csv(config['raw_text_path'], sep='\t')
    node2paper = pd.read_csv('dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz')
    train_idx = pd.read_csv('dataset/ogbn_arxiv/split/time/train.csv.gz', header=None)
    valid_idx = pd.read_csv('dataset/ogbn_arxiv/split/time/valid.csv.gz', header=None)
    test_idx = pd.read_csv('dataset/ogbn_arxiv/split/time/test.csv.gz', header=None)
    label = pd.read_csv('dataset/ogbn_arxiv/raw/node-label.csv.gz', header=None)
    
    # Preprocess & modify csv error
    raw_data.columns = ['Id', 'Title', 'Abstract']
    raw_data.iloc[0, 0] = 200971
    raw_data = raw_data.drop(len(raw_data)-1)

    train_idx = train_idx.iloc[:, 0].tolist()
    valid_idx = valid_idx.iloc[:, 0].tolist()
    test_idx = test_idx.iloc[:, 0].tolist()
    idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    label = label.iloc[:, 0].tolist()

    # Create node_id->paper_id dict, paper_id->node_id dict
    paper2node_dict = {}
    node2paper_dict = {}
    for i, row in tqdm(node2paper.iterrows()):
        paper2node_dict[int(row[1])] = int(row[0])
        node2paper_dict[int(row[0])] = int(row[1])
    
    train, valid, test = tokenize(raw_data, paper2node_dict, idx, label)

    with open(config['train'], 'wb') as f:
        pickle.dump(train, f)
    with open(config['valid'], 'wb') as f:
        pickle.dump(valid, f)
    with open(config['test'], 'wb') as f:
        pickle.dump(test, f)
    with open(config['paper2node'], 'wb') as f:
        pickle.dump(paper2node_dict, f)
    with open(config['node2paper'], 'wb') as f:
        pickle.dump(node2paper_dict, f)
    
if __name__ == "__main__":
    main()

