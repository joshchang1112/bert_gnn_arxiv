import pickle

def load_data(model_type):
    """ Load dataset preprocess by tokenizer."""
    with open('dataset/ogbn_arxiv/{}/train.pkl'.format(model_type), 'rb') as f:
        train = pickle.load(f)
    with open('dataset/ogbn_arxiv/{}/val.pkl'.format(model_type), 'rb') as f:
        valid = pickle.load(f)
    with open('dataset/ogbn_arxiv/{}/test.pkl'.format(model_type), 'rb') as f:
        test = pickle.load(f)
    return train, valid, test

def pad_to_len(arr, padded_len, padding=0):
    length_arr = len(arr)
    new_arr = arr
    if length_arr < padded_len:
        for i in range(padded_len - length_arr):
            new_arr.append(padding)
    else:
        for i in range(length_arr - padded_len):
            del new_arr[-2]
    return new_arr

