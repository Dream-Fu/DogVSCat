import argparse
import torch as t
import utils.utils as utils
from torch.utils.data import DataLoader
from datasets.CatVSDog import DogCat
from models.resnet import ResNet34
from torch import optim
import torch.nn as nn
import os
import shutil

parser = argparse.ArgumentParser()
# 数据集路径
# parser.add_argument('--file_path', type=str, default= './data/train/', help='whether to train.txt')
parser.add_argument('--train_path', type=str, default= './data/train/', help='whether to train.txt')
parser.add_argument('--val_path', type=str, default= '/home/aries/Downloads/datasets/expression/lists/valshuffle.txt', help='whether to val.txt')
# 模型及数据存储路径
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='directory where model checkpoints are saved')
# 网络选择
parser.add_argument('--model', type=str, default='resnet',help='which net is chosen for training ')
# 批次
parser.add_argument('--batch_size', type=int, default=10, help='size of each image batch')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
# cuda设置
parser.add_argument('--cuda', type=str, default="0", help='whether to use cuda if available')
# CPU载入数据线程设置
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
# 暂停设置
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
# 迭代次数
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
# 起始次数（针对resume设置）
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
# 显示结果的间隔
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
# 确认参数，并可以通过opt.xx的形式在程序中使用该参数
opt = parser.parse_args()

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    train_dataset = DogCat(opt.train_path, train=True, test=False)
    val_dataset = DogCat(opt.train_path, train=False, test=False)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    if opt.model == 'resnet':
        model = ResNet34()

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    best_precision = 0
    lowest_loss = 10000
    for epoch in range(opt.epochs):
                                            # (train_loader, model, criterion, optimizer, epoch, print_interval, filename):
        acc_train, loss_train = utils.train(train_loader, model, criterion, optimizer, epoch, opt.print_interval, opt.checkpoint_dir)
        # 在日志文件中记录每个epoch的训练精度和损失
        with open(opt.checkpoint_dir+'each_epoch_record.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, train_Precision: %.8f, train_Loss: %.8f\n' % (epoch, acc_train, loss_train))
        # 测试
        precision, avg_loss = utils.validate(val_loader, model, criterion, optimizer, epoch, opt.print_interval, opt.checkpoint_dir)
        # 在日志文件中记录每个epoch的验证精度和损失
        with open(opt.checkpoint_dir + 'each_epoch_record_val.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))
            pass

        # 记录最高精度与最低loss
        best_precision = max(precision, best_precision)
        lowest_loss = min(avg_loss, lowest_loss)
        print('--' * 30)
        print(' * Accuray {acc:.3f}'.format(acc=precision),
              '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss:.3f}'.format(loss=avg_loss),
              'Previous Lowest Loss: %.3f)' % lowest_loss)
        print('--' * 30)
        # 保存最新模型
        save_path = os.path.join(opt.checkpoint_dir, 'checkpoint.pth')
        t.save(model.state_dict(), save_path)
        # 保存准确率最高的模型
        is_best = precision > best_precision
        is_lowest_loss = avg_loss < lowest_loss

        if is_best:
            best_path = os.path.join(opt.checkpoint_dir, 'best_model.pth')
            shutil.copyfile(save_path, best_path)
        # 保存损失最低的模型
        if is_lowest_loss:
            lowest_path = os.path.join(opt.checkpoint_dir, 'lowest_loss.pth')
            shutil.copyfile(save_path, lowest_path)




