# import DANN
import torch
import torch.optim as optim
import tqdm
import mydataset
import argparse  # 参数设置
# 引用python库要规范，不然会出现错误
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import prototype_test
# import global_match
import random
from torch.backends import cudnn
import csv
import os

parser = argparse.ArgumentParser()  # 创建对象

# 添加参数
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nepoch', type=int, default=300)
parser.add_argument('--acc_start', type=int, default=90)  # 调参 80-85-90-95
# parse_args()将之前add_argument()定义的参数进行赋值，并返回相关的设置
args = parser.parse_args()

# 选择运算设备
DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


# DEVICE = torch.device('cpu')

def test(model, target_dataloader, acc):
    model.eval()
    correct = 0
    with torch.no_grad():
        for target in target_dataloader:
            data_t, label_t, _ = target

            label_t = label_t.long()

            data_t, label_t = data_t.to(DEVICE), label_t.to(DEVICE)

            class_output = model('test', acc=acc, data_t=data_t, acc_start=args.acc_start)
            class_output = torch.argmax(class_output, dim=1)
            correct = correct + (class_output == label_t).sum().item()

    acc = float(correct) / len(target_dataloader.dataset) * 100
    return acc


'''
dataloader_src:source data
dataloader_tar:target data
'''


def train(model, optimizer, source_dataloader, target_dataloader, index, total, num_class, p_index, person_id, dataset):
    epoch_acc = 0
    lengh = min(len(source_dataloader), len(target_dataloader))
    print(len(source_dataloader))
    print(len(target_dataloader))
    epoch_per_person = torch.ones([args.nepoch, total])  # 如果是其他数据集需要修改
    train_result = ['train_acc']
    test_result = ['test_acc']
    path = 'F:/code-HBW/model_1/result/%s parameter_%d%d%d%d%d/'%(dataset,
            p_index[0], p_index[1], p_index[2], p_index[3], p_index[4])

    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(args.nepoch):
        model.train()  # 训练声明(启用 BatchNormalization 和 Dropout)
        correct = 0
        batch_per_person = torch.ones([lengh, total])  # 如果是其他数据集需要修改
        # tqdm：设置进度条
        for batch, (source, target) in enumerate(zip(source_dataloader, target_dataloader)):

            data_s, label_e, person = source
            data_t, _, _ = target

            # p_1 = label_e.sum()/len(label_e)
            # p_0 = 1 - p_1
            # print('train',p_1.item(),'***',p_0.item())

            label_s = torch.zeros(data_s.size(0))
            label_t = torch.ones(data_t.size(0))

            label_e = label_e.long()
            label_s = label_s.long()
            label_t = label_t.long()
            person = person.long()

            data_s, data_t, label_e, label_s, label_t, person = data_s.to(DEVICE), data_t.to(DEVICE), \
                                                                label_e.to(DEVICE), label_s.to(DEVICE), \
                                                                label_t.to(DEVICE), person.to(DEVICE)

            # alpha小trick
            p = (100 - epoch_acc) / 100
            alpha = 1 - np.exp(-p)
            # p = float(batch + 1 + epoch * lengh) / args.nepoch / lengh
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # e_output = model('train', label_e, data_s, data_t, alpha, DEVICE)

            e_output, domain_s, domain_t, iso_s, iso_t, weight = model('train', epoch_acc, label_e, data_s, data_t,
                                                                       alpha, args.acc_start)

            for j in index:
                # 直接等于为浅拷贝，原值一同变化！！！
                person_ite = person.clone()

                if j == 0:
                    person_ite[person_ite != j] = -1
                    person_ite = person_ite + 1
                else:
                    person_ite[person_ite != j] = 0
                    person_ite[person_ite == j] = 1

                w = (person_ite * weight).sum()
                batch_per_person[batch, j] = w

            err_s = 0
            err_t = 0
            # 计算loss
            if epoch_acc > args.acc_start:
                gamma = 0.6
                # for i in range(num_class):
                #     err_si = F.nll_loss(F.log_softmax(domain_s[i], dim=1), label_s,
                #                         reduction='none')
                #     err_si = (err_si * weight).mean()
                #     err_ti = F.nll_loss(F.log_softmax(domain_t[i], dim=1), label_t)
                #     err_s += err_si
                #     err_t += err_ti
                err_s = F.nll_loss(F.log_softmax(domain_s, dim=1), label_s, reduction='none')
                err_s = (err_s * weight).mean()
                err_t = F.nll_loss(F.log_softmax(domain_t, dim=1), label_t)
                err_e = F.nll_loss(F.log_softmax(e_output, dim=1), label_e, reduction='none')
                err_e = (err_e * weight).mean()
            else:
                gamma = 0.4
                err_s = F.nll_loss(F.log_softmax(domain_s, dim=1), label_s)
                err_t = F.nll_loss(F.log_softmax(domain_t, dim=1), label_t)
                err_e = F.nll_loss(F.log_softmax(e_output, dim=1), label_e)

            err_d = err_s + err_t

            err_iso_s = F.nll_loss(F.log_softmax(iso_s, dim=1), label_s)
            err_iso_t = F.nll_loss(F.log_softmax(iso_t, dim=1), label_t)
            err_iso = err_iso_s + err_iso_t

            err = err_e + gamma * err_d + err_iso
            optimizer.zero_grad()  # 梯度值清零
            err.backward()  # 计算梯度值
            optimizer.step()  # 梯度下降参数更新
            e_output = torch.argmax(e_output, dim=1)
            correct = correct + (e_output == label_e).sum().item()

        epoch_per_person[epoch, :] = batch_per_person.sum(dim=0)

        # item返回tensor的元素值
        epoch_acc = float(correct) / len(source_dataloader.dataset) * 100
        # item_pr = 'Train Epoch: [{}/{}], emotion_loss: {:.4f}, domain_loss : {:.4f} total_loss: {:.4f}, Epoch{}_Acc: {:.4f}' \
        #     .format(epoch, args.nepoch, err_e.item(), err_d.item(), err.item(), epoch, epoch_acc)
        item_pr = 'Train Epoch: [{}/{}], emotion_loss: {:.4f}, domain_loss: {:.4f}, iso_loss: {:.4f}, total_loss: {:.4f}, Epoch{}_Acc: {:.4f}' \
            .format(epoch, args.nepoch, err_e.item(), err_d.item(), err_iso.item(), err.item(), epoch, epoch_acc)
        print(item_pr)
        train_result.append(epoch_acc)
        # 记录训练结果(https://blog.csdn.net/u011389474/article/details/60140311?ops_request_misc=&request_id=&biz_id=102&utm_term=fp%20=%20open(args.result_path,%20%27a&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-60140311.first_rank_v2_pc_rank_v29)
        fp = open(path +'person_%d train_process.txt' %(person_id), 'a')
        fp.write(item_pr + '\n')
        fp.close()

        # 记录测试结果
        test_acc = test(model, target_dataloader, epoch_acc)
        test_result.append(test_acc)
        test_info = 'Test acc Epoch{}: {:.4f}'.format(epoch, test_acc)
        print(test_info)

        fp = open(path +'person_%d train_process.txt' %(person_id), 'a')
        fp.write(test_info + '\n')
        fp.close()

    epoch_index = torch.arange(0, args.nepoch, 1).reshape(-1, 1)
    epoch_per_person = torch.cat((epoch_index, epoch_per_person), dim=1)
    epoch_per_person = epoch_per_person.detach().numpy()

    csvFile = open(path + 'train_test_person_%d.csv' %(person_id), "a", newline='')
    writer = csv.writer(csvFile)
    writer.writerow(train_result)
    writer.writerow(test_result)
    csvFile.close()

    np.savetxt(path + 'person_%d attention.csv' %(person_id), epoch_per_person,
               delimiter=',', fmt='%.03f')
    print('save down')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    # 初始化
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # torch.cuda.manual_seed_all(1)

    dataset = ['DEAP', 'DREAMER', 'SEED']
    exp = ['SEED_cross_subject', 'SEED_cross_session']
    sample_num = [32, 14, 15]

    mode = dataset[2]  # DEAP/DREAMER/SEED
    SEED_mode = exp[0]
    sample_path = 'E:/%s/leave one subject out/' % (mode)

    if mode == 'SEED':
        label = 'label.npy'  # label_V/label_A
        channel = 62
        total_sample = sample_num[2]
        num_class = 3
        in_dim = 100
        if SEED_mode == exp[1]:
            total_session = 3
    elif mode == 'DREAMER':
        total_sample = sample_num[1]
        label = 'label_V.npy'  # label_V/label_A
        channel = 14
        num_class = 2
        in_dim = 128 * 5
    elif mode == 'DEAP':
        total_sample = sample_num[0]
        label = 'label_V.npy'  # label_V/label_A
        channel = 32
        num_class = 2
        in_dim = 128

    # parameter

    hid_dim_1 = [256, 512]
    hid_dim_2 = [64, 128, 256, 512]
    out_dim = [64, 128, 256, 512]
    kernel_size = [4, 8, 16]
    dim = [512, 1024, 2048]

    total = total_sample

    for p_1 in range(len(hid_dim_1)):
        for p_2 in range(len(hid_dim_2)):
            for p_3 in range(len(out_dim)):
                for p_4 in range(len(kernel_size)):
                    for p_5 in range(len(dim)):
                        p_index = [p_1, p_2, p_3, p_4, p_5]
                        print(p_index)
                        for i in range(total):
                            person_id = i
                            torch.cuda.empty_cache()
                            print('*' * 20)
                            print('target subject:', i)
                            print('*' * 20)
                            target_sample = np.load(sample_path + 'person_%d data.npy' % (i))
                            if mode == 'SEED':
                                e_label =np.load(sample_path + label)
                                target_elabel = np.tile(e_label, 3)
                            else:
                                target_elabel = np.load(sample_path + 'person_%d ' % (i) + label)
                            person_t = np.ones(len(target_elabel)) * i

                            if SEED_mode == exp[1]:
                                source_sample = target_sample[15:1080, :, :]
                                source_elabel = target_elabel[15:1080]
                                target_sample = target_sample[0:15, :, :]
                                target_elabel = target_elabel[0:15]
                                person_t = np.zeros(15)
                                person_s = np.append(np.ones_like(person_t))
                                total = total_session
                                index = [1]

                            else:

                                index = [k for k in range(total)]
                                del index[i]
                                print('source index:', index)

                                # p_1 = target_elabel.sum() / len(target_elabel)
                                # p_0 = 1 - p_1
                                # print('target', p_1.item(), '***', p_0.item())

                                for k, j in enumerate(index):
                                    if k == 0:
                                        source_sample = np.load(sample_path + 'person_%d data.npy' % (j))
                                        if mode == 'SEED':
                                            source_elabel = target_elabel
                                        else:
                                            source_elabel = np.load(sample_path + 'person_%d ' % (j) + label)
                                        person_s = np.ones([len(source_elabel)]) * j
                                    else:
                                        data = np.load(sample_path + 'person_%d data.npy' % (j))
                                        # data = data[0:15, :, :]
                                        if mode == 'SEED':
                                            elabel = target_elabel
                                        else:
                                            elabel = np.load(sample_path + 'person_%d ' % (j) + label)
                                        source_sample = np.append(source_sample, data, axis=0)
                                        source_elabel = np.append(source_elabel, elabel)
                                        person_s = np.append(person_s, np.ones([len(elabel)], dtype=int) * j)

                            # p_1 = source_elabel.sum() / len(source_elabel)
                            # p_0 = 1 - p_1
                            # print('source', p_1.item(), '***', p_0.item())
                            ratio = len(source_elabel) // len(target_elabel)

                            print(len(source_elabel))
                            print('*' * 20)
                            print(source_sample.shape)
                            print('*' * 20)
                            print(len(target_elabel))
                            print('*' * 20)
                            print(target_sample.shape)

                            source = mydataset.Mydataset(source_sample, source_elabel, person_s)

                            target = mydataset.Mydataset(target_sample, target_elabel, person_t)

                            source_dataloader = DataLoader(source, batch_size=10 * ratio, shuffle=True)

                            target_dataloader = DataLoader(target, batch_size=10, shuffle=True)

                            '''
                            hid_dim_1 = [64, 128, 256, 512]
                            hid_dim_2 = [64, 128, 256, 512]
                            out_dim = [64, 128, 256, 512]
                            kernel_size = [4, 8, 16]
                            dim = [512, 1024, 2048]
                            '''

                            model = prototype_test.Model(hid_dim_1=hid_dim_1[p_index[0]],
                                                         out_dim_1=out_dim[p_index[1]],
                                                         hid_dim_2=hid_dim_2[p_index[2]],
                                                         out_dim_2=channel,
                                                         hid_dim_3=hid_dim_1[p_index[0]],
                                                         out_dim_3=out_dim[p_index[1]],
                                                         hid_dim_4=hid_dim_2[p_index[2]],
                                                         channel=channel,
                                                         num_class=num_class,
                                                         in_dim=in_dim,
                                                         kernel_size=kernel_size[p_index[3]],
                                                         dim=dim[p_index[4]]).to(DEVICE)
                            # model.load_state_dict(torch.load(args.model_path))

                            optimizer = optim.SGD(model.parameters(), lr=args.lr)

                            print('train begin')

                            train(model, optimizer, source_dataloader, target_dataloader, index, total, num_class, p_index, person_id, mode)

                            total = total_sample
