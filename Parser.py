
import json
import string
from collections import defaultdict

import numpy as np
import torch

PAD = '<pad>'
UNIQUE = '<unk>'
NUMBER = '<num>'  # just prepare if need to parse by words


class Parser:

    def __init__(self, filename, F2I={}, L2I={}, length=None):
        self.glov = {}
        self.glov[UNIQUE] = np.zeros(300)
        self.glove_dict()
        self.file_name = filename
        self.length = length
        vocab, self.sentences, self.labels, self.max_sentence = self.Parse()
        self.F2I = F2I if F2I else self.create_dict(vocab)
        self.L2I = L2I if L2I else self.create_dict(self.labels, should_pad=False)
        self.indexer_glov()
        self.indexer_labels()
        # self.indexer_sentences()
        self.sentence_padding()

    def glove_dict(self):
        with open("./Data/glove.6B.300d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.glov[word] = vector

    def Parse(self):
        data = []
        labels = []
        vocab = set()
        vocab.add(UNIQUE)
        max_sentence_len = -1
        wordFreqDict = defaultdict(lambda: 0)
        file = open(self.file_name, 'r')
        for line in file:
            loader = json.loads(line)
            sent_1 = self.Parse_Line(loader['sentence1'], wordFreqDict, vocab)
            sent_2 = self.Parse_Line(loader['sentence2'], wordFreqDict, vocab)
            gold_label = loader['gold_label']
            if gold_label == '-':
                continue
            labels.append(gold_label)
            if len(sent_1) > max_sentence_len or len(sent_2) > max_sentence_len:
                max_sentence_len = max(len(sent_1), len(sent_2))
            data.append([sent_1, sent_2])
        #rare words
        for i in range(len(data)):
            data[i][0] = [w if wordFreqDict[w] > 1 else UNIQUE for w in data[i][0]]
            data[i][1] = [w if wordFreqDict[w] > 1 else UNIQUE for w in data[i][1]]

        return vocab, data, labels, max_sentence_len

    def Parse_Line(self, line, wordFreqDict, vocab):
        punctuation =set(string.punctuation)
        new_line =list()
        for word in line.split(' '):
            word = word.lower()
            word = ''.join(ch for ch in word if ch not in punctuation)
            new_line.append(word)
            vocab.add(word)
            wordFreqDict[word] += 1
        return new_line

    @staticmethod
    def create_dict(vocab, should_pad=True):
        data_dict = {f: i for i, f in enumerate(list(sorted(set(vocab))))}
        if should_pad:
            data_dict[list(data_dict.keys())[0]] = len(data_dict)
            data_dict[PAD] = 0
        return data_dict

    def get_F2I(self):
        return self.F2I

    def get_L2I(self):
        return self.L2I

    def get_Data(self):
        return self.sentences

    def get_Labels(self):
        return self.labels

    def get_max_word_len(self):
        max_len = 0
        for word in self.F2I.keys():
            if len(word) > max_len:
                max_len = len(word)
        return max_len
    def get_glov(self):
        return self.glov

    def indexer_glov(self):
        for pair in self.sentences:
            for index, word in enumerate(pair[0]):
                if pair[0][index] in self.glov:
                    pair[0][index] = self.glov[word]
            for index, word in enumerate(pair[1]):
                if pair[1][index] in self.glov:
                    pair[1][index] = self.glov[word]

    def indexer_sentences(self):
        for pair in self.sentences:
            for index, word in enumerate(pair[0]):
                if pair[0][index] in self.F2I:
                    pair[0][index] = self.F2I[word]
                else:
                    pair[0][index] = self.F2I[UNIQUE]
            for index, word in enumerate(pair[1]):
                if pair[1][index] in self.F2I:
                    pair[1][index] = self.F2I[word]
                else:
                    pair[1][index] = self.F2I[UNIQUE]

    def indexer_labels(self):
        for index, label in enumerate(self.labels):
            self.labels[index] = self.L2I[label]

    def sentence_padding(self):
        word_index = self.glov[UNIQUE]
        for pair in self.sentences:
            while len(pair[0]) < self.max_sentence:
                pair[0].append(word_index)
            while len(pair[1]) < self.max_sentence:
                pair[1].append(word_index)
        if self.length is not None:
            for index, pair in enumerate(self.sentences):
                self.sentences[index] = [pair[0][:self.length], pair[1][:self.length]]

    def DataLoader(self,batch_size, shuffle=True):
        sent_1 = []
        sent_2 = []
        for sentence in self.sentences:
            sent_1.append(sentence[0])
            sent_2.append(sentence[1])
        batchs = [((sent_1[i * batch_size: (i + 1) * batch_size], sent_2[i * batch_size: (i + 1) * batch_size]),
                    self.labels[i * batch_size: (i + 1) * batch_size])
                   for i in range(int(len(sent_1) / batch_size))]
        batchs.append(((sent_1[-(len(sent_1) % batch_size):], sent_2[-(len(sent_2) % batch_size):]),
                        self.labels[-(len(self.labels) % batch_size):]))

        for index, batch in enumerate(batchs):
          #batchs[index] = ((torch.LongTensor(np.asarray(batch[0][0])), torch.LongTensor(np.asarray(batch[0][1]))), torch.LongTensor(np.asarray(batch[1])))
          batchs[index] = ((torch.LongTensor(np.asarray(batch[0][0])), torch.LongTensor(np.asarray(batch[0][1]))),
                           torch.LongTensor(np.asarray(batch[1])))

        return batchs
