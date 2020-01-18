import re
import json

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

class Parser:
    PAD = '<pad>'
    UNIQUE = '<uuukkk>'
    NUMBER = '<num>'

    def __init__(self, filename):
        self.file_name = filename

    def Parse(self):
        data = []
        max_len = -1
        file = open(self.file_name,'r')
        for line in file:
            loader = json.loads(line)
            sent_1 = loader['sentence1']
            sent_2 = loader['sentence2']
            gold_label = loader['gold_label']
            if len(sent_1) > max_len or len(sent_2)>max_len:
                max_len = max(len(sent_1), len(sent_2))
            data.append((sent_1,sent_2, gold_label))
        return data, max_len


if __name__ == '__main__':
    p = Parser('./Data/snli_1.0_train.jsonl')
    dataset, max_len = p.Parse()
    print("hi")

