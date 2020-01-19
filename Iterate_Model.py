
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


def train(model, encoder, train_set, model_optimizer, encoder_optimizer, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    sum_examples = len(train_set)
    model.train()
    for sentences, labels in train_set:
        model_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        encoded = encoder(sentences[0], sentences[1])
        predictions = model(encoded[0], encoded[1])
        loss = loss_fn(predictions, labels)
        epoch_acc += calc_batch_accuracy(predictions, labels)
        epoch_loss += loss
        loss.backward()
        encoder_optimizer.step()
        model_optimizer.step()
    return float(epoch_loss) / sum_examples, float(epoch_acc) / sum_examples, model


def evaluate(model, encoder, dev_set, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    sum_examples = len(dev_set)
    model.eval()
    for sentences, labels in dev_set:
        encoded = encoder(sentences[0], sentences[1])
        predictions = model(encoded[0], encoded[1])
        loss = loss_fn(predictions, labels)
        epoch_acc += calc_batch_accuracy(predictions, labels)
        epoch_loss += loss
    return float(epoch_loss) / sum_examples, float(epoch_acc) / sum_examples


def iterate_model(model, encoder, train_set, dev_set, lr=0.01, epochs=10):
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss = nn.NLLLoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, model = train(model, encoder, train_set, model_optimizer, encoder_optimizer, loss)
        dev_loss, dev_acc = evaluate(model, encoder, dev_set, loss)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tDev Loss: {dev_loss:.3f} | Dev Acc: {dev_acc * 100:.2f}%')
    return model


def main():
    train_parser = Parser('./Data/snli_1.0_train.jsonl')
    F2I = train_parser.get_F2I()
    L2I = train_parser.get_L2I()
    dev_parser = Parser('./Data/snli_1.0_dev.jsonl', F2I, L2I)
    test_parser = Parser('./Data/snli_1.0_test.jsonl', F2I, L2I)
    embedding_dim = 300
    hidden_dim = 1000
    batch_size = 20
    output_dim = 3
    encoder = Encode(len(F2I), embedding_dim, hidden_dim)
    model = SelfAttention(hidden_dim, output_dim, dropout_rate=0.2)


if __name__ == "__main__":
    main()
