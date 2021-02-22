# Node classification with fine-tuned BERT encoder &amp; GNN (Pytorch)

This repository contains a pytorch implementation of fine-tuned BERT encoder & GNN 
for `ogbn-arxiv` node classification.

If you did not have the GPU environment to run this code, you can check the [**colab tutorial**](https://colab.research.google.com/github/joshchang1112/bert_gnn_arxiv/blob/master/pytorch/fine_tuned_bert_gnn_pytorch.ipynb) or [**github notebook**](https://github.com/joshchang1112/bert_gnn_arxiv/blob/master/pytorch/fine_tuned_bert_gnn_pytorch.ipynb) that will allow you to get start faster:)

## Installation

To execute our code successfully, you need to install Python3 and PyTorch (our deep learning framework) first. Please refer to [Python installing page](https://www.python.org/downloads/) and [Pytorch installing page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, transformers and pytorch_geometric can be installed using pip as follows:
```
pip install ogb
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-geometric
pip install transformers
```

## Code organization

*   `download_dataset.sh`: Download `ogbn-arxiv` graph dataset and the raw texts of titles and abstracts in MAG [[1]](#references) papers. 

*   `config.json`: Store the file path of data, fine-tuned model and node features.

*   `make_bert_dataset.py`: Tokenize the raw texts of papers, convert word tokens to ids and split dataset for below training using BERT.

*   `fine_tuned_bert.py`: Fine-tuned BERT model on multi-classification task with args parameters. `python fine_tuned_bert.py --help` for more information. 

*   `dataset.py`: The dataset when fine-tuning BERT model.

*   `encode_features.py`: Encode the node features.

*   `gnn.py`: Train Graph Neural Networks(GNN) on `ogbn-arxiv` dataset.

*   `metrics.py`: Calculate accuracy for training.

*   `utils.py`: Data processing utils functions.

## Code usage

1.  Download `ogbn-arxiv` dataset.
```
bash download_dataset.sh
```

2.  Preprocess the data and split the dataset for training.
```
python3 make_bert_dataset.py
```

3.  Fine-tuned BERT on ogbn-arxiv with default parameters. (Test Accuracy: ~72.1%)
```
python3 fine_tuned_bert.py 
```

4.  Use fine-tuned BERT model to encode the raw texts of titles and abstracts to node features.
```
python3 encode_features.py
```

5. Train GCN on ogbn-arxiv using fine-tuned BERT node features. (Test Accuracy: ~74.9%)
```
python3 gnn.py --num_layers=2 --lr=5e-3 --bert_features=True
```

You can also run `python3 gnn.py` to train GCN on ogbn-arxiv using the embeddings pretrained by skip-gram model [[2]](#references) which provided from Open Graph Benchmark. (Test Accuracy: ~71.8%)

If you do not want to train BERT model by yourself, or you are only interested in how GNN works powerfully by using fine-tuned BERT node features, you can just download the model and node features trained in the above method using GeForce GTX 1080 Ti. 

Download Link: [Fine-tuned BERT](https://www.dropbox.com/s/6zlln7alz0mtmy4/fine-tuned_bert_10.pkl?dl=0), [BERT node features](https://www.dropbox.com/s/gbw1rl2ayxnl91q/bert_feat_10.pkl?dl=0)

Furthermore, you can just run `sh run.sh` to reproduce our results. If you want to see the details of results, `log.txt` records the accuracy about valid and test set in different seeds.

Finally, it is very excited that the results by **our proposed method even beats the 1st place in Open Graph Benchmark Leaderboard**! We encourage users to experiment further by trying different architecture to encode features and using other advanced GNN models for multi-class node classification on `ogbn-arxiv` dataset.


## References

[1] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representationsof words and phrases and their compositionality. In Advances in Neural Information Processing Systems (NeurIPS), pp. 3111–3119, 2013.



