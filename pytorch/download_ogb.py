from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
import torch_geometric.transforms as T
dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
