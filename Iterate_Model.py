
import torch
import torch.nn as nn


def calc_batch_accuracy(predictions, labels, idx_to_label):
    correct = wrong = 0
    for pred, label in zip(predictions, labels):
        if pred.argmax() == label:
            if not idx_to_label[int(label)] == 'O':
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


def train(model, train_set, optimizer, loss_fn, idx_to_label):
    epoch_loss = 0
    epoch_acc = 0
    sum_examples = len(train_set)
    model.train()
    for windows, labels in train_set:
        optimizer.zero_grad()
        predictions = model(windows)
        loss = loss_fn(predictions, labels)
        epoch_acc += calc_batch_accuracy(predictions, labels, idx_to_label)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    return float(epoch_loss) / sum_examples, float(epoch_acc) / sum_examples, model


def evaluate(model, dev_set, loss_fn, idx_to_label):
    epoch_loss = 0
    epoch_acc = 0
    sum_examples = len(dev_set)
    model.eval()
    for windows, labels in dev_set:
        predictions = model(windows)
        loss = loss_fn(predictions, labels)
        epoch_acc += calc_batch_accuracy(predictions, labels, idx_to_label)
        epoch_loss += loss
    return float(epoch_loss) / sum_examples, float(epoch_acc) / sum_examples


def trainer_loop(model, train_set, dev_set, idx_to_label, lr=0.01, epochs=10, msg=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss, train_acc, model = train(model, train_set, optimizer, loss, idx_to_label)
        dev_loss, dev_acc = evaluate(model, dev_set, loss, idx_to_label)
        print('Epoch: ' + str(epoch + 1))
        print(f'\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc * 100:2f}%')
        print(f'\tDev Loss: {dev_loss:.3f}, Dev Acc: {dev_acc * 100:2f}%')
    return model
