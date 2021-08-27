import model1
import torch
import torch.optim as optim
import tqdm
import mydataset
import argparse  # 参数设置
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
"""
SEED数据集模型
输入的总样本数为675
数据格式为310*180
目标：三分类（积极、中立、消极）
标签为0、1、2
domain classifier有三个
"""

parser = argparse.ArgumentParser()  # 创建对象

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nEpoch', type=int, default=200)
# parser.add_argument('--result_path', type=str, default='G:/emotion recognition_DAAN_2
# parser.add_argument('--data_path', type=str, default='G:/DREAMER/leave one subject out/index')

# parse_args()将之前add_argument()定义的参数进行赋值，并返回相关的设置
args = parser.parse_args()
# 选择运算设备
DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


def test(myNet, test_data_loader):
    alpha = 0
    myNet.eval()
    correct = 0
    with torch.no_grad():
        for t_sample, t_label in tqdm.tqdm(test_data_loader):
            t_sample, t_label = t_sample.cuda(), t_label.cuda()
            class_output, _, _ = myNet('test', t_sample, t_sample, alpha)
            class_output = torch.argmax(class_output, dim=1)
            correct = correct + (class_output == t_label).sum().item()
    acc = float(correct) / len(test_data_loader.dataset) * 100
    return acc


def train(myNet, optimizer, source_data_loader, target_data_loader):
    best_acc = -float('inf')
    length = min(len(source_data_loader), len(target_data_loader))
    train_accuracy = []
    test_accuracy = []

    for epoch in range(args.nEpoch):
        myNet.train()
        correct = 0

        # tqdm：设置进度条
        for (source, target) in tqdm.tqdm(zip(source_data_loader, target_data_loader), total=length, leave=True):

            data_s, label_s_e = source[0].cuda(), source[1].cuda()
            data_t, label_t_e = target[0].cuda(), target[1].cuda()
            # 属于源域的样本被设置为0
            label_s_d = torch.zeros(data_s.size(0))
            # 属于目标域的样本被设置为1
            label_t_d = torch.ones(data_t.size(0))
            # alpha小trick
            # p = float(length + epoch * length) / args.nEpoch / length
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 1
            s_e_output, domain_s, domain_t = myNet('train', data_s, data_t, alpha)
            # 计算loss
            label_s_e = label_s_e.long()
            label_s_d = label_s_d.long()
            label_t_d = label_t_d.long()

            err_s = 0
            err_t = 0
            err_e = F.nll_loss(F.log_softmax(s_e_output, dim=1), label_s_e)
            for j in range(3):
                err_si = F.nll_loss(F.log_softmax(domain_s[j].cuda(), dim=1), label_s_d.cuda())
                err_ti = F.nll_loss(F.log_softmax(domain_t[j].cuda(), dim=1), label_t_d.cuda())
                err_s += err_si
                err_t += err_ti
            err_d = err_s + err_t
            err = err_e + err_d
            optimizer.zero_grad()  # 梯度值清零
            err.backward()  # 计算梯度值
            optimizer.step()  # 梯度下降参数更新
            scheduler.step()
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            e_output = torch.argmax(s_e_output, dim=1)

            correct = correct + (e_output == label_s_e).sum().item()

        # item返回tensor的元素值
        train_acc = float(correct) / (len(source_data_loader.dataset)) * 100
        train_acc1 = round(train_acc, 4)
        train_accuracy.append(train_acc1)
        item_pr = 'Train Epoch: [{}/{}], emotion_loss: {:.4f}, ' \
                  'domain_loss: {:.4f} total_loss: {:.4f}, epoch{}_Acc: {:.4f}' \
            .format(epoch, args.nEpoch, err_e.item(), err_d.item(), err.item(), epoch, train_acc)
        print(item_pr)

        # 记录测试结果
        test_acc = test(myNet, target_data_loader)
        test_acc1 = round(test_acc, 4)
        test_accuracy.append(test_acc1)
        test_info = 'Test acc Epoch{}: {:.4f}'.format(epoch, test_acc)
        print(test_info)
        if best_acc < test_acc:
            best_acc = test_acc
        # torch.save(model.state_dict(), args.model_path)
        best_info = 'best_Test_acc: {:.4f}'.format(best_acc)
        # fp = open(args.result_path, 'a')
        # fp.write(Best_info + '\n')
        # fp.close()
    print(best_info)
    print(train_accuracy)
    print(test_accuracy)
    x = []
    for i in range(args.nEpoch):
        x.append(i)
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax.plot(x, train_accuracy)
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.plot(x, test_accuracy)
    plt.show()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print("="*120+'\n'+"cuda is available!"+'\n'+"="*120)
    mode = 'SEED'  # DEAP/DREAMER
    sample_path = 'F:/Alex/SEED/leave one subject out/segment/'
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

        # p = int(source_sample.shape[0] // target_sample.shape[0])
        # target_sample = np.repeat(target_sample, axis=0, repeats=p)
        # target_elabel = np.repeat(target_elabel, axis=0, repeats=p)

        source = mydataset.Mydataset(source_sample, source_elabel)
        target = mydataset.Mydataset(target_sample, target_elabel)
        source_data_loader = DataLoader(source, batch_size=70, shuffle=True, drop_last=True)
        target_data_loader = DataLoader(target, batch_size=5, shuffle=True, drop_last=True)
        lr_list = []
        myNet = model1.NewModel(in_channel=310, out_channel=128, res_inchannel=256, res_outchannel=128).cuda()
        LR = 0.1
        optimizer = optim.SGD(myNet.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 150], gamma=0.9)
        train(myNet, optimizer, source_data_loader, target_data_loader)