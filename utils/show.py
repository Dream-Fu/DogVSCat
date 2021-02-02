import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    train_file = open("../checkpoint/record_iter_train.txt", 'r')
    train_total = train_file.readlines()
    train_num = len(train_total)
    train_res = np.zeros(train_num)
    train_idx = np.arange(train_num)
    acc_list = []
    loss_list = []

    for idx in range(train_num):
        train_str = train_total[idx].split(',')
        train_acc = float(train_str[0].split(':')[-1])
        acc_list.append(train_acc)
        train_loss = float(train_str[1].split(':')[-1])
        loss_list.append(train_loss)

    plt.figure()
    plt.title('train_acc')
    plt.plot(train_idx, acc_list)
    # plt.legend('batch')
    plt.savefig('acc.png')
    plt.show()

    plt.figure()
    plt.title('train_loss')
    plt.plot(train_idx, loss_list)
    plt.savefig('loss.png')
    plt.show()

