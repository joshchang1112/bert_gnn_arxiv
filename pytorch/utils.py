import pickle
import os
import json

def load_data(config_path):
    """ Load dataset preprocess by tokenizer."""
    with open(config_path) as f:
        config = json.load(f)
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)
    with open(config['valid'], 'rb') as f:
        valid = pickle.load(f)
    with open(config['test'], 'rb') as f:
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

