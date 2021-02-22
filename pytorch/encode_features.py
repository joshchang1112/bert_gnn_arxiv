from transformers import BertTokenizer, BertModel
import pickle
import argparse
import json
import pandas as pd
import torch
import os
from tqdm import tqdm

def freeze_bert_layers(model):
    """Freeze all bert layers to release GPU memory"""
    freeze_layers = 12
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    model.bert.embeddings.dropout.p = 0.0
    for p in model.bert.pooler.parameters():
        p.requires_grad = False
    for idx in range(freeze_layers):
        for p in model.bert.encoder.layer[idx].parameters():
            p.requires_grad = False
        model.bert.encoder.layer[idx].attention.self.dropout.p = 0.0
        model.bert.encoder.layer[idx].attention.output.dropout.p = 0.0
        model.bert.encoder.layer[idx].output.dropout.p = 0.0
    return model

def encode_features(data, data_len, paper2node_dict, model):
    node_feats = torch.zeros((data_len, 768)).cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = freeze_bert_layers(model)
    for i, row in tqdm(data.iterrows()):
        if row['Id'] not in paper2node_dict:
            continue
        context = "[CLS] " + row['Title'] + row['Abstract'] + " [SEP]"
        tokenize_context = tokenizer.tokenize(context)
        context_len = len(tokenize_context)

        if context_len > 512:
            tokenize_context = tokenize_context[:512]
    
        context_id = tokenizer.convert_tokens_to_ids(tokenize_context)
        context_id = torch.LongTensor(context_id).unsqueeze(0).cuda()
        feat = model.bert(context_id)[0].squeeze(0)[0]
        node_id = paper2node_dict[row['Id']]
        node_feats[node_id, :] = feat
        torch.cuda.empty_cache()
    
    return node_feats
    

def main():
    parser = argparse.ArgumentParser(description='Encode Node Features')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--node_feat_dir', type=str, default='node_feat',
                        help='Directory to the fine-tuned node features.')
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available()
                          else 'cpu')

    with open('config.json') as f:
        config = json.load(f)
    
    # Load data & model
    MODEL_PATH = config['bert_models'].format(args.seed)
    data = pd.read_csv(config['raw_text_path'], sep='\t')
    with open(config['node2paper'], 'rb') as f:
        node2paper_dict = pickle.load(f)
    with open(config['paper2node'], 'rb') as f:
        paper2node_dict = pickle.load(f)
    
    data.columns = ['Id', 'Title', 'Abstract']
    data.iloc[0, 0] = 200971
    data = data.drop(len(data)-1)
    model = torch.load(MODEL_PATH).to(device)
    
    # Create or check directory
    if os.path.isdir(args.node_feat_dir) == False:
        os.makedirs(args.node_feat_dir)
        print('Create folder: {}'.format(args.node_feat_dir))
    else:
        print('{} exists!'.format(args.node_feat_dir))

    node_feats = encode_features(data, len(node2paper_dict),
                                 paper2node_dict, model)

    torch.save(node_feats, config['node_features'].format(args.seed))

if __name__ == '__main__':
    main()
