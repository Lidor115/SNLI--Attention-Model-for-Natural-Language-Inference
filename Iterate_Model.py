import numpy as np
import torch
import torch.nn as nn
import time
from Model import Encode, SelfAttention
from Parser import Parser


def calc_batch_accuracy(predictions, labels):
    correct = wrong = 0
    for pred, label in zip(predictions, labels):
        if pred.argmax() == label:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


def predict(model, windows, file_type, L2I, msg):
    with open('./data/test_{0}.txt'.format(msg), mode='w') as file:
        predictions = list()
        for window in windows:
            y = model(window[0])
            _, y = torch.max(y, 1)
            y = L2I[int(y)]
            predictions.append(y)
            file.write("{0}\n".format(y))
    file.close()
    return predictions


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds

def encoded_sentences(sentences, F2I):
    index_sent = 0
    sentences_ret = []
    for sent in sentences:
        sentence=[]
        index_word =0
        for word in sent:
            sentence.append(F2I[word].reshape(1,-1))
            index_word +=1
        sentence = torch.cat(sentence)
        sentence =(sentence).reshape(1,sentence.shape[0], sentence.shape[1])
        sentences_ret.append(sentence)
        index_sent+=1
    return torch.cat(sentences_ret)



def train(model, train_set, model_optimizer, loss_fn, I2F,F2I):
    epoch_loss = 0
    epoch_acc = 0
    sum_examples = len(train_set)
    model.train()
    for sentences, labels in train_set:
        model_optimizer.zero_grad()
        #encoder_optimizer.zero_grad()
        encoded_0 = (encoded_sentences(sentences[0],F2I))
        encoded_1 = (encoded_sentences(sentences[1],F2I))
        predictions = model(encoded_0,encoded_1)
        loss = loss_fn(predictions, torch.LongTensor(labels))
        epoch_acc += calc_batch_accuracy(predictions, labels)
        epoch_loss += loss
        loss.backward()
        model_optimizer.step()
        #encoder_optimizer.step()
    return float(epoch_loss) / sum_examples, float(epoch_acc) / sum_examples, model


def evaluate(model, dev_set, loss_fn,I2F,F2I):
    epoch_loss = 0
    epoch_acc = 0
    sum_examples = len(dev_set)
    model.eval()
    for sentences, labels in dev_set:
        encoded_0 = (encoded_sentences(sentences[0],F2I))
        encoded_1 = (encoded_sentences(sentences[1],F2I))
        predictions = model(encoded_0,encoded_1)
        loss = loss_fn(predictions, torch.LongTensor(labels))
        epoch_acc += calc_batch_accuracy(predictions, labels)
        epoch_loss += loss
    return float(epoch_loss) / sum_examples, float(epoch_acc) / sum_examples


def iterate_model(model, train_set, dev_set,I2F,F2I, lr=0.01, epochs=10):
    print('hi')
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, model = train(model, train_set, model_optimizer, loss,I2F,F2I)
        dev_loss, dev_acc = evaluate(model, dev_set, loss,I2F,F2I)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tDev Loss: {dev_loss:.3f} | Dev Acc: {dev_acc * 100:.2f}%')
    return model


def main():
    train_parser = Parser('./Data/snli_1.0_train.jsonl', length=25)
    F2I = train_parser.get_F2I()
    L2I = train_parser.get_L2I()
    dev_parser = Parser('./Data/snli_1.0_dev.jsonl', F2I, L2I, length=25)
    #test_parser = Parser('./Data/snli_1.0_test.jsonl', F2I, L2I, length=25)
    embedding_dim = 300
    hidden_dim = 200
    batch_size = 10
    output_dim = 3
    I2F = train_parser.get_glov_T()
    F2I = train_parser.get_glov()
    #encoder = Encode(len(F2I), embedding_dim, hidden_dim)
    model = SelfAttention(embedding_dim,hidden_dim, output_dim)
    iterate_model(model, train_parser.DataLoader(batch_size, shuffle=True),
                  dev_parser.DataLoader(batch_size, shuffle=True),I2F,F2I)


if __name__ == "__main__":
    main()
