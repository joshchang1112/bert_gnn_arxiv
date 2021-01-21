import torch
from torch.utils.data import Dataset
from utils import pad_to_len

class CitationDataset(Dataset):

    def __init__(self, data, max_length, padding=0):
        self.data = data
        self.max_seq_len = max_length
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        if len(data['context']) > self.max_seq_len:
            data['context'] = data['context'][:self.max_seq_len]
        return data

    def collate_fn(self, datas):
        batch = {}
        batch['length'] = torch.LongTensor([data['length'] for data in datas])
        padded_len = min(self.max_seq_len, max(batch['length']))
        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas]
        )
        batch['label'] = torch.LongTensor([data['label'] for data in datas])
        return batch
