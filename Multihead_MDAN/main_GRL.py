from MultiResolution.Multihead_MDAN import model_transferable_attention
import torch
import torch.optim as optim
import tqdm
import MultiResolution.mydataset as dataset
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
"""
SEED数据集模型
输入的总样本数为675
数据格式为62*900
目标：三分类（积极、中立、消极）
标签为0、1、2
segment6:数据形式(45, 310, 180),对(45, 62, 180, 5)的dim=180进行标准化,然后再进行拼接成(45, 62, 900)
segment7:数据处理方式为moving average; 数据形式(45, 310, 180),对(45, 62, 180, 5)的dim=180进行标准化,然后再进行拼接成(45, 62, 900)

"""

parser = argparse.ArgumentParser()  # 创建对象
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nEpoch', type=int, default=100)

# parse_args()将之前add_argument()定义的参数进行赋值，并返回相关的设置
args = parser.parse_args()
# 选择运算设备


def test(myNet, test_data_loader):
    alpha = 0
    myNet.eval()
    correct = 0
    with torch.no_grad():
        for t_sample, t_label in tqdm.tqdm(test_data_loader):
            t_sample, t_label = t_sample.cuda(), t_label.cuda()
            output = myNet(t_sample, t_sample, alpha)
            class_output = output[0]
            class_output = torch.argmax(class_output, dim=1)
            correct = correct + torch.eq(class_output, t_label).float().sum().item()
    acc = correct / len(test_data_loader.dataset) * 100
    return acc


def train(myNet, source_loader, target_loader):
    length = len(source_loader)
    torch.cuda.manual_seed(100)
    lr = 0.003
    learning_rate = lr / math.pow(1 + 10 * (epoch-1) / args.nEpoch, 0.75)
    myNet.train()
    correct = 0
    i = 0
    for source_data, source_label in tqdm.tqdm(source_loader, desc='Train epoch={}'.format(epoch), total=length, leave=True):
        # training model using source data
        # (args.nEpoch * length)
        p = float((i + epoch * length) / args.nEpoch / length)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        data_s, label_s_e = source_data.cuda(), source_label.cuda()
        for target_data, target_label in target_loader:
            data_t, label_t_e = target_data.cuda(), target_label.cuda()
            break
        label_s_d = torch.zeros(data_s.size(0))
        label_t_d = torch.ones(data_t.size(0))
        # use the model to train
        out = myNet(data_s, data_t, alpha)
        s_pred = out[0]
        local_domain_s = out[1]
        local_domain_t = out[2]
        bottle_feature_s_reshape = out[3]
        bottle_feature_t_reshape = out[4]
        label_s_e = label_s_e.long()
        label_s_d = label_s_d.long()
        label_t_d = label_t_d.long()
        # definite the loss function
        criteon = nn.CrossEntropyLoss()
        loss_weight = [1, 0.01, 0.001]
        # Batch spectral penalization loss
        _, s_s, _ = torch.linalg.svd(bottle_feature_s_reshape, full_matrices=False)
        _, s_t, _ = torch.linalg.svd(bottle_feature_t_reshape, full_matrices=False)
        sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        # print(s_s,s_t)
        # class loss
        loss_e = criteon(s_pred.cuda(), label_s_e.cuda())
        # local domain loss
        loss_s = 0
        loss_t = 0
        for j in range(3):
            loss_si = criteon(local_domain_s[j].cuda(), label_s_d.cuda())
            loss_ti = criteon(local_domain_t[j].cuda(), label_t_d.cuda())
            loss_s += loss_si
            loss_t += loss_ti
        loss_d = loss_s + loss_t
        # total loss
        loss_total = loss_weight[0] * loss_e + loss_weight[1] * loss_d + loss_weight[2] * sigma
        loss_total.backward()
        optimizer.step()
        i += 1
        e_output = torch.argmax(s_pred, dim=1)
        # classification accuracy
        correct += torch.eq(e_output, label_s_e).float().sum().item()
    train_acc = correct / (len(source_loader.dataset)) * 100
    train_acc1 = round(train_acc, 2)
    train_accuracy.append(train_acc1)
    item_pr = 'Train Epoch: [{}/{}], emotion_loss: {:.2f}, ' \
              'domain loss: {:.2f}, sigma loss:{:.2f},total_loss: {:.2f}, epoch{}_Acc: {:.2f}' \
        .format(epoch, args.nEpoch, loss_e.item(), loss_d.item(), sigma.item(), loss_total.item(), epoch, train_acc)
    print(item_pr)

    # collect data
    err_e1 = loss_e.item()
    error_e = round(err_e1, 2)
    e_error.append(error_e)
    err_d1 = loss_d.item()
    error_d = round(err_d1, 2)
    d_error.append(error_d)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print("="*120+'\n'+"cuda is available!"+'\n'+"="*120)
    # load data
    mode = 'SEED'  # DEAP/DREAMER
    sample_path = 'G:/Alex/SEED_experiment/leave one subject out/segment7/'
    # label = 'label_V.npy'
    # label_V/label_A此处用以更换情绪标签类别
    if mode == 'DREAMER':
        total = 23
    elif mode == 'DEAP':
        total = 32
    elif mode == 'SEED':
        total = 15
    for i in range(total):
        print('*' * 20)
        print('target subject:', i)
        print('*' * 20)
        #  第i个作为target domain
        #  数据格式为bach*channel*length
        target_sample = np.load(sample_path+'person_%d data.npy' % i)
        target_elabel = np.load(sample_path+'label.npy')
        # 生成列表list
        index = [k for k in range(total)]
        # 删除第i个的元素
        del index[i]
        #  划分源域
        for k, j in enumerate(index):
            if k == 0:
                source_sample = np.load(sample_path+'person_%d data.npy' % j)
                source_elabel = np.load(sample_path+'label.npy')

            else:
                data = np.load(sample_path+'person_%d data.npy' % j)
                elabel = np.load(sample_path+'label.npy')
                source_sample = np.append(source_sample, data, axis=0)
                #  最终数据形式为bach_size*channel*length
                source_elabel = np.append(source_elabel, elabel)
        # increase target data
        p = int(source_sample.shape[0] // target_sample.shape[0])
        target_sample = np.repeat(target_sample, axis=0, repeats=p)
        target_elabel = np.repeat(target_elabel, axis=0, repeats=p)
        # construct dataset
        source = dataset.Mydataset(source_sample, source_elabel)
        target = dataset.Mydataset(target_sample, target_elabel)
        source_loader = DataLoader(source, batch_size=16, shuffle=True, drop_last=False)
        target_loader = DataLoader(target, batch_size=16, shuffle=True, drop_last=False)
        myNet = model_transferable_attention.OverallModel().cuda()
        optimizer = optim.SGD(myNet.parameters(), lr=0.001, momentum=0.0)
        train_accuracy = []
        test_accuracy = []
        e_error = []
        d_error = []
        best_acc = -float('inf')
        for epoch in range(1, args.nEpoch+1):
            train(myNet, source_loader, target_loader)
            test_acc = test(myNet, target_loader)
            test_acc1 = round(test_acc, 2)
            test_accuracy.append(test_acc1)
            test_info = 'Test acc Epoch{}: {:.2f}'.format(epoch, test_acc)
            print(test_info)
            if best_acc < test_acc:
                best_acc = test_acc
            best_info = 'best_Test_acc: {:.2f}'.format(best_acc)
        print(best_info)
        print('train_accuracy:', train_accuracy)
        print('test_accuracy', test_accuracy)
        print('error_e:', e_error)
        print('error_d:', d_error)

        # draw the picture of result
        x = []
        for i in range(1, args.nEpoch+1):
            x.append(i)
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)
        ax.plot(x, train_accuracy)
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax1.plot(x, test_accuracy)
        ax2.plot(x, e_error)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('error_e')
        ax3.plot(x, d_error)
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('error_d')
        iter = torch.rand(1)
        path = 'G:/Alex/SEED_experiment/results/multidomain_GRL/' + str(iter) + '.jpg'
        plt.savefig(path)
