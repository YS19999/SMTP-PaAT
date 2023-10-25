
import re
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.utils import tprint

def _load_csv(path):
    label = {}
    text_len = []

    dataset = pd.read_csv(path, encoding='UTF-8')
    texts, intents = dataset['content'], dataset['label']

    data = []
    for i, line in enumerate(texts):

        if int(intents[i]) not in label:
            label[int(intents[i])] = 1
        else:
            label[int(intents[i])] += 1

        item = {
            'text': line,
        }

        text_len.append(len(line))
        data.append(item)

    tprint('dataset: {}, sample number: {}'.format(path, len(data)))

    return data

def stop_word_replace(data):

    stopword = []
    id2word = {}

    replace_word = open('data/EnglishStopWords1.txt', 'r', encoding='utf-8')

    for i, line in enumerate(replace_word):
        stopword.append(line.rstrip('\n').lstrip('\ufeff'))
        id2word[int(i)] = line.rstrip('\n').lstrip('\ufeff')

    replace_word.close()

    for i, line in enumerate(data):
        sent = []
        for word in re.findall(r'\w+|[^\w\s]', line['text']):
            if word in stopword:
                random_ids = random.randint(0, len(id2word)-1)
                word = id2word[random_ids]
            sent.append(word)
        data[i] = {'text':' '.join(sent)}

    return data

def _del_by_idx(array_list, idx, axis):

    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list

def _data_to_nparray(data, tokenizer):

    for e in data:
        tokens = tokenizer(e['text'], max_length=60, truncation=True, padding='max_length', return_tensors="pt")
        e['bert_id'], e['attn_mask'], e['token_type_ids'] = tokens['input_ids'][0].numpy(), tokens['attention_mask'][0].numpy(), tokens['token_type_ids'][0].numpy()

    text_len = np.array([len(e['bert_id']) for e in data])
    max_text_len = max(text_len)

    text = np.zeros([len(data), max_text_len], dtype=np.int64)
    text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)
    token_type_ids = np.zeros([len(data), max_text_len], dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']
        text_mask[i, :len(data[i]['attn_mask'])] = data[i]['attn_mask']
        token_type_ids[i, :len(data[i]['token_type_ids'])] = data[i]['token_type_ids']

        # filter out document with only special tokens
        # unk (100), cls (101), sep (102), pad (0)
        if np.max(text[i]) < 103:
            del_idx.append(i)

    text_len, text = _del_by_idx([text_len, text], del_idx, 0)

    new_data = {
        'text': torch.tensor(text),
        'attn_mask': torch.tensor(text_mask),
        'token_type_ids': torch.tensor(token_type_ids)
    }

    return new_data

class Samplers(Dataset):
  
    def __init__(self, tokenizer, data1, data2, data3):

        self.data1 = [tokenizer(text['text'],
                                padding='max_length',
                                max_length=60,
                                truncation=True,
                                return_tensors="pt") for text in data1]
        self.data2 = [tokenizer(text['text'],
                                padding='max_length',
                                max_length=60,
                                truncation=True,
                                return_tensors="pt") for text in data2]
        self.data3 = [tokenizer(text['text'],
                                padding='max_length',
                                max_length=60,
                                truncation=True,
                                return_tensors="pt") for text in data3]

    def __len__(self):
        return len(self.data1)

    def get_batch_data1(self, item):
        return self.data1[item]

    def get_batch_data2(self, item):
        return self.data2[item]

    def get_batch_data3(self, item):
        return self.data3[item]

    def __getitem__(self, item):
        return self.get_batch_data1(item), self.get_batch_data2(item), self.get_batch_data3(item)

def shuffle(lst):
    temp_lst = deepcopy(lst)
    n = m = len(temp_lst)
    while m:
        m -= 1
        i = random.randint(0, n - 1)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst


def data_loader(args, tokenizer, data1, data2, data3):

    train_dataset = Samplers(tokenizer, data1, data2, data3)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader

def loader_dataset(args, tokenizer):

    tprint('data loader...')

    data1 = []

    for name in args.dataname:
        path = 'data/' + name + '.csv'
        data1 += _load_csv(path)

    tprint('data total number: {}'.format(len(data1)))

    tprint('for data augmentation')
    data2 = data3 = data1.copy()
    data2 = stop_word_replace(data2)

    data3 = shuffle(data3)

    tprint('get dataloader')
    dataloader = data_loader(args, tokenizer, data1, data2, data3)

    return dataloader

