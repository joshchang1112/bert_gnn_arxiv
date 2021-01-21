python3 download_ogb.py
wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
gunzip titleabs.tsv.gz
mv titleabs.tsv dataset/ogbn_arxiv/raw
