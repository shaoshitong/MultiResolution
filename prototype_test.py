import torch.nn as nn
import torch
import GRL
import copy
'''
prototype
input:b*32*640
output:b*2
feature_extract -> classification -> source prototype(average) ->
metric -> target classification -> class-wised adaptation ->
'''

# MLP_Mix
class MLP_Mix(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP_Mix, self).__init__()
        # self.DEVICE = torch.device("cuda")
        self.MLP = nn.Sequential()
        self.MLP.add_module('l1', nn.Linear(in_dim, hid_dim))
        self.MLP.add_module('gelu1', nn.GELU())
        self.MLP.add_module('l2', nn.Linear(hid_dim, out_dim))

    def forward(self, x):

        x = self.MLP(x)
        layerNorm = nn.LayerNorm(x.size()[1:]).cuda()
        x = layerNorm(x)
        return x


class Model(nn.Module):

    def __init__(self, hid_dim_1, out_dim_1, hid_dim_2, out_dim_2,
                 hid_dim_3, out_dim_3, hid_dim_4, channel, num_class,
                 in_dim, kernel_size, dim):
        super(Model, self).__init__()

        self.softmax = nn.Softmax(dim=1)
        self.num_class = num_class

        # MLP
        self.feature_dim = int(channel * ((out_dim_3 - kernel_size) / (kernel_size/2) + 1))

        self.MLP_1 = MLP_Mix(in_dim, hid_dim_1, out_dim_1)
        self.MLP_2 = MLP_Mix(channel, hid_dim_2, out_dim_2)
        self.MLP_3 = MLP_Mix(out_dim_1, hid_dim_3, out_dim_3)
        self.MLP_4 = MLP_Mix(out_dim_2, hid_dim_4, channel)
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=int(kernel_size/2))

        # label classification
        self.L = nn.Sequential()
        self.L.add_module('l1', nn.Linear(self.feature_dim, self.num_class))
        # self.L.add_module('relu1', nn.ReLU(True))
        # self.L.add_module('l2', nn.Linear(512, 2))

        # domain classification
        # self.local_D = nn.Sequential()
        # self.dci = {}
        # for i in range(self.num_class):
        #     self.dci[i] = nn.Sequential()
        #     self.dci[i].add_module('fc1', nn.Linear(self.feature_dim, 2048))
        #     self.dci[i].add_module('relu1', nn.ReLU(True))
        #     self.dci[i].add_module('fc2', nn.Linear(2048, 1024))
        #     self.dci[i].add_module('relu2', nn.ReLU(True))
        #     self.dci[i].add_module('fc3', nn.Linear(1024, 2))
        #     # self.dci[i].add_module('relu3', nn.ReLU(True))
        #     # self.dci[i].add_module('fc4', nn.Linear(256, 2))
        #     self.local_D.add_module('dci_' + str(i), self.dci[i])

        self.Global_D = nn.Sequential()
        self.Global_D.add_module('fc1', nn.Linear(self.feature_dim, dim))
        self.Global_D.add_module('relu1', nn.ReLU(True))
        self.Global_D.add_module('fc2', nn.Linear(dim, int(dim/2)))
        self.Global_D.add_module('relu2', nn.ReLU(True))
        self.Global_D.add_module('fc3', nn.Linear(int(dim/2), 2))
        # self.Global_D.add_module('relu3', nn.ReLU(True))
        # self.Global_D.add_module('fc4', nn.Linear(512, 2))

        self.D_weight = nn.Sequential()
        self.D_weight.add_module('fc1', nn.Linear(self.feature_dim, 2))
        # self.D_weight.add_module('relu2', nn.ReLU(True))
        # self.D_weight.add_module('fc3', nn.Linear(512, 2))

    def forward(self, mode, acc=None, label_e=None, data_s=None, data_t=None, alpha=None, acc_start=None):

        if mode == 'train':

            b_s, c, _ = data_s.shape
            b_t, _, _ = data_t.shape

            # 如果精度上升到90以上可以理解为提取的特征是有效的，使用sample-weight
            if acc > acc_start:

                # feature extraction
                data_s = self.MLP_1(data_s)
                data_s = data_s.permute(0, 2, 1)
                data_s = self.MLP_2(data_s)
                data_s = data_s.permute(0, 2, 1)
                data_s = self.MLP_3(data_s)
                data_s = data_s.permute(0, 2, 1)
                data_s = torch.squeeze(self.MLP_4(data_s))
                data_s = self.pool(data_s.permute(0, 2, 1))
                feature_s = torch.flatten(data_s, 1)  # 20,2016

                data_t = self.MLP_1(data_t)
                data_t = data_t.permute(0, 2, 1)
                data_t = self.MLP_2(data_t)
                data_t = data_t.permute(0, 2, 1)
                data_t = self.MLP_3(data_t)
                data_t = data_t.permute(0, 2, 1)
                data_t = torch.squeeze(self.MLP_4(data_t))
                data_t = self.pool(data_t.permute(0, 2, 1))
                feature_t = torch.flatten(data_t, 1)  # 20,2016

                # sample_reweight (1) 为了更好的学习权重高的样本
                iso_feature_s = GRL.Iso_layer.apply(feature_s)
                iso_s_out = self.D_weight(iso_feature_s)
                iso_s_label = self.softmax(iso_s_out)

                iso_feature_t = GRL.Iso_layer.apply(feature_t)
                iso_t_out = self.D_weight(iso_feature_t)

                weight = 1 / (iso_s_label[:,0] / iso_s_label[:,1] + 1).cuda()
                # weight = torch.unsqueeze(weight,1)

                # 最大最小归一化
                weight = (weight - weight.min())/(weight.max()-weight.min())

                print(weight.mean())
                # feature_s = feature_s * weight

                # label classification
                s_label = self.L(feature_s)

                reverse_feature_s = GRL.ReverseLayerF.apply(feature_s, alpha)
                reverse_feature_t = GRL.ReverseLayerF.apply(feature_t, alpha)

                # Global_GRL
                s_out = self.Global_D(reverse_feature_s)
                t_out = self.Global_D(reverse_feature_t)

                # SEED prototype计算
                # num = torch.ones(3).cuda()
                # label_cls = torch.ones([self.num_class, b_s]).cuda()
                # for i in range(self.num_class):
                #     label_copy = copy.deepcopy(label_e)
                #     if i != 0:
                #         label_copy[label_copy != i] = 0
                #         label_copy[label_copy == i] = 1
                #     else:
                #         label_copy[label_copy != i] = -1
                #         label_copy += 1
                #     label_cls[i] = label_copy
                #
                # prototype_0 = torch.mul(feature_s, label_cls[0].reshape(-1, 1)).sum(dim=0) / num[0]
                # prototype_1 = torch.mul(feature_s, label_cls[1].reshape(-1, 1)).sum(dim=0) / num[1]
                # prototype_2 = torch.mul(feature_s, label_cls[2].reshape(-1, 1)).sum(dim=0) / num[2]
                # t_label = torch.ones([b_t, self.num_class]).cuda()
                #
                # # 欧氏距离，距离越近则差异越小
                # d_0 = feature_t - prototype_0
                # d_1 = feature_t - prototype_1
                # d_2 = feature_t - prototype_2
                # for i in range(b_t):
                #     dis_0 = -torch.sqrt((torch.dot(d_0[i, :], d_0[i, :]) / self.feature_dim))
                #     dis_1 = -torch.sqrt((torch.dot(d_1[i,:],d_1[i,:])/self.feature_dim))
                #     dis_2 = -torch.sqrt((torch.dot(d_2[i, :], d_2[i, :]) / self.feature_dim))
                #     t_label[i, 0] = dis_0
                #     t_label[i, 1] = dis_1
                #     t_label[i, 2] = dis_2

                # 余弦相似度 数值越大则差异越小
                # dot_1 = torch.sqrt(torch.dot(prototype_1, prototype_1))
                # dot_0 = torch.sqrt(torch.dot(prototype_0, prototype_0))
                # for i in range(b_t):
                #     dis_1 = torch.dot(feature_t[i, :], prototype_1) / \
                #             dot_1 / torch.sqrt(torch.dot(feature_t[i, :], feature_t[i, :]))
                #     dis_0 = torch.dot(feature_t[i, :], prototype_0) / \
                #             dot_0 / torch.sqrt(torch.dot(feature_t[i, :], feature_t[i, :]))
                #     t_label[i, 0] = dis_0
                #     t_label[i, 1] = dis_1

                # target = self.softmax(t_label)
                # source = self.softmax(s_label)
                #
                # s_out = []
                # t_out = []
                #
                # for i in range(self.num_class):
                #     # 维度转换变成可以用来进行sub-domain分类的维度
                #     ps = source[:, i].reshape((b_s, 1))
                #     fs = ps * reverse_feature_s
                #     pt = target[:, i].reshape((b_t, 1))
                #     ft = pt * reverse_feature_t
                #     outsi = self.local_D[i](fs)
                #     s_out.append(outsi)
                #     outti = self.local_D[i](ft)
                #     t_out.append(outti)

            # 初步提取域不变特征
            else:

                # feature extraction
                data_s = self.MLP_1(data_s)
                data_s = data_s.permute(0, 2, 1)
                data_s = self.MLP_2(data_s)
                data_s = data_s.permute(0, 2, 1)
                data_s = self.MLP_3(data_s)
                data_s = data_s.permute(0, 2, 1)
                data_s = torch.squeeze(self.MLP_4(data_s))
                data_s = self.pool(data_s.permute(0, 2, 1))
                feature_s = torch.flatten(data_s, 1)  # 20,2016

                data_t = self.MLP_1(data_t)
                data_t = data_t.permute(0, 2, 1)
                data_t = self.MLP_2(data_t)
                data_t = data_t.permute(0, 2, 1)
                data_t = self.MLP_3(data_t)
                data_t = data_t.permute(0, 2, 1)
                data_t = torch.squeeze(self.MLP_4(data_t))
                data_t = self.pool(data_t.permute(0, 2, 1))
                feature_t = torch.flatten(data_t, 1)  # 20,2016

                # weight
                iso_feature_s = GRL.Iso_layer.apply(feature_s)
                iso_s_out = self.D_weight(iso_feature_s)
                iso_s_label = self.softmax(iso_s_out)

                iso_feature_t = GRL.Iso_layer.apply(feature_t)
                iso_t_out = self.D_weight(iso_feature_t)

                weight = 1 / (iso_s_label[:,0] / iso_s_label[:,1] + 1).cuda()
                # weight = torch.unsqueeze(weight,1)
                weight = (weight - weight.min())/(weight.max()-weight.min())
                print(weight.mean())

                # label classification
                s_label = self.L(feature_s)

                # GRL
                reverse_feature_s = GRL.ReverseLayerF.apply(feature_s, alpha)
                reverse_feature_t = GRL.ReverseLayerF.apply(feature_t, alpha)

                s_out = self.Global_D(reverse_feature_s)
                t_out = self.Global_D(reverse_feature_t)

            return s_label, s_out, t_out, iso_s_out, iso_t_out, weight

        else:

            # # feature extraction
            # data_s = self.MLP_1(data_s)
            # data_s = data_s.permute(0, 2, 1)
            # data_s = self.MLP_2(data_s)
            # data_s = data_s.permute(0, 2, 1)
            # data_s = self.MLP_3(data_s)
            # data_s = data_s.permute(0, 2, 1)
            # data_s = torch.squeeze(self.MLP_4(data_s))
            # data_s = self.pool(data_s.permute(0, 2, 1))
            # feature_s = torch.flatten(data_s, 1)  # 20,2016

            data_t = self.MLP_1(data_t)
            data_t = data_t.permute(0, 2, 1)
            data_t = self.MLP_2(data_t)
            data_t = data_t.permute(0, 2, 1)
            data_t = self.MLP_3(data_t)
            data_t = data_t.permute(0, 2, 1)
            data_t = torch.squeeze(self.MLP_4(data_t))
            data_t = self.pool(data_t.permute(0, 2, 1))
            feature_t = torch.flatten(data_t, 1)  # 20,2016

            # num_1 = label_e.sum()
            # num_0 = b_s - num_1

            # # prototype计算
            # label_e = torch.unsqueeze(label_e, 1)
            # prototype_1 = torch.mul(feature_s, label_e).sum(dim=0)/num_1
            # prototype_0 = torch.mul(feature_s, (label_e - 1).abs()).sum(dim=0)/num_0
            # t_label = torch.ones([b_t, 2]).cuda()
            #
            # # 欧氏距离
            # d_1 = feature_t - prototype_1
            # d_0 = feature_t - prototype_0
            # for i in range(b_t):
            #     dis_1 = -torch.sqrt((torch.dot(d_1[i,:],d_1[i,:])/self.feature_dim))
            #     dis_0 = -torch.sqrt((torch.dot(d_0[i,:],d_0[i,:])/self.feature_dim))
            #     t_label[i, 0] = dis_0
            #     t_label[i, 1] = dis_1

            # # 余弦相似度
            # dot_1 = torch.sqrt(torch.dot(prototype_1, prototype_1))
            # dot_0 = torch.sqrt(torch.dot(prototype_0, prototype_0))
            # for i in range(b_t):
            #     dis_1 = 1 + torch.dot(feature_t[i, :], prototype_1) / \
            #             dot_1 / torch.sqrt(torch.dot(feature_t[i, :], feature_t[i, :]))
            #     dis_0 = 1 + torch.dot(feature_t[i, :], prototype_0) / \
            #             dot_0 / torch.sqrt(torch.dot(feature_t[i, :], feature_t[i, :]))
            #     t_label[i, 0] = dis_0
            #     t_label[i, 1] = dis_1
            t_label = self.L(feature_t)

            return t_label