import re
import json
import string
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

PAD = '<pad>'
UNIQUE = '<unk>'
NUMBER = '<num>'  # just prepare if need to parse by words


class Parser:

    def __init__(self, filename):
        self.file_name = filename
        self.vocab = set()

    def Parse(self):
        data = list()
        max_len = -1
        wordFreqDict = defaultdict(lambda: 0)
        file = open(self.file_name,'r')
        for line in file:
            loader = json.loads(line)
            sent_1 = self.Parse_Line(loader['sentence1'],wordFreqDict)
            sent_2 = self.Parse_Line(loader['sentence2'],wordFreqDict)
            gold_label = loader['gold_label']
            if len(sent_1) > max_len or len(sent_2)>max_len:
                max_len = max(len(sent_1), len(sent_2))
            data.append([sent_1,sent_2, gold_label])
        for i in range(len(data)):
            data[i][0]= [w if wordFreqDict[w] >20 else UNIQUE for w in data[i][0]]
            data[i][1]= [w if wordFreqDict[w] >20 else UNIQUE for w in data[i][1]]





        return data, max_len

    def Parse_Line(self, line,wordFreqDict):
        punctuation =set(string.punctuation)
        new_line =list()
        for word in line.split(' '):
            word = word.lower()
            word = ''.join(ch for ch in word if ch not in punctuation)
            new_line.append(word)
            wordFreqDict[word] += 1
        return new_line





