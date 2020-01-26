import csv

import numpy as np
from matplotlib import pyplot as plt


def plot_graphs(dev_acc_list, dev_loss_list, iters, name):
    add_to_name = " accuracy.png"
    ticks = int(iters / 10)
    if not ticks:
        ticks = 1
    plt.plot(range(iters + 1), dev_acc_list)
    plt.xticks(np.arange(0, iters + 1, step=1))
    plt.yticks(np.arange(0, 110, step=10))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('{} accuracy'.format(name))
    for i in range(0, len(dev_acc_list), ticks):
        #plt.annotate(round(dev_acc_list[i], 1), (i, dev_acc_list[i]))
        plt.savefig(name + add_to_name)
    plt.close()
    #plt.show()

    plt.plot(range(iters + 1), dev_loss_list)
    plt.xticks(np.arange(0, iters + 1, step=1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('{} loss'.format(name))
    for i in range(0, len(dev_loss_list), ticks):
        #plt.annotate(round(dev_loss_list[i], 2), (i, dev_loss_list[i]))
        add_to_name = " loss.png"
        plt.savefig(name + add_to_name)
    #plt.show()
    plt.close()



def parseCsv(file):

    with open (file, 'r') as file:
        train_acc = []
        dev_acc = []
        dev_loss = []
        train_loss = []
        epochs =[]
        name = "GloVe 50"
        reader = csv.reader(file)
        for row in reader:
            epochs.append(int(row[0]))
            dev_acc.append(float(row[1]))
            dev_loss.append(float(row[2]))
            train_acc.append(float(row[3]))
            train_loss.append(float(row[4]))
        plot_graphs(dev_acc,dev_loss,epochs[-1],name + " - test set")
        plot_graphs(train_acc,train_loss,epochs[-1],name + " - train set")


if __name__ == '__main__':
    parseCsv('Data/GloVe 50.csv')
