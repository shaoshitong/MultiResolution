import torch.nn as nn
import torch

"""
SEED数据集模型
输入的总样本数为675
数据格式为310*180
目标：三分类（积极、中立、消极）
标签为0、1、2
domain classifier有三个
"""


class ChannelWiseAttention(nn.Module):
    #  设置in_channel=32或14或310
    def __init__(self, in_channel, reduction=16):
        super(ChannelWiseAttention, self).__init__()
        self.Mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel//reduction),
            nn.ReLU(),
            nn.Linear(in_channel//reduction, in_channel),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AvgPool1d(kernel_size=180)
        self.max_pool = nn.MaxPool1d(kernel_size=180)

    def forward(self, input_data):
        # 输入的是三维数据B*C*L
        # 第i个通道求平均池化（[B, C, L]）
        avg_pool = torch.squeeze(self.avg_pool(input_data))
        avg_out = self.Mlp(avg_pool)
        # 第i个通道求最大池化（[B, C, 1]）
        max_pool = torch.squeeze(self.max_pool(input_data))
        max_out = self.Mlp(max_pool)
        # 叠加求scale
        out = avg_out+max_out
        scale = out.view(input_data.shape[0], input_data.shape[1], 1)
        data_out = scale * input_data
        return data_out


class LowResolutionCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 设置in_channel=310,out_channel=128
        super(LowResolutionCNN, self).__init__()
        self.LowResolution = nn.Sequential()
        self.LowResolution.add_module('L_conv1', nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=3))
        self.LowResolution.add_module('L_relu1', nn.ReLU())
        self.LowResolution.add_module('L_pool1', nn.AvgPool1d(kernel_size=8, stride=2))
        self.LowResolution.add_module('L_conv2', nn.Conv1d(out_channel, 2*out_channel, kernel_size=2, stride=2))
        self.LowResolution.add_module('L_relu2', nn.ReLU())
        # self.LowResolution.add_module('L_conv3', nn.Conv1d(2*out_channel, 2*out_channel, kernel_size=2, stride=2))
        # self.LowResolution.add_module('L_relu3',  nn.ReLU())  # (256,13)

    def forward(self, x):
        x = self.LowResolution(x)
        return x


class HighResolutionCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 设置in_channel=310,out_channel=128
        super(HighResolutionCNN, self).__init__()
        self.HighResolution = nn.Sequential()
        self.HighResolution.add_module('H_conv1', nn.Conv1d(in_channel, out_channel, kernel_size=20, stride=3))
        self.HighResolution.add_module('H_relu1', nn.ReLU())
        self.HighResolution.add_module('H_pool1', nn.AvgPool1d(kernel_size=4, stride=2))
        self.HighResolution.add_module('H_conv2', nn.Conv1d(out_channel, 2*out_channel, kernel_size=2, stride=2))
        self.HighResolution.add_module('H_relu2', nn.ReLU())
        # self.HighResolution.add_module('H_conv3', nn.Conv1d(2*out_channel, 2*out_channel, kernel_size=2, stride=2))
        # self.HighResolution.add_module('H_relu3', nn.ReLU())  # (256,13)

    def forward(self, x):
        x = self.HighResolution(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, res_inchannel, res_outchannel):
        #  输入为B*256*L,res_inchannel=256, res_outchannel=128
        super(ResidualBlock, self).__init__()
        self.residualNet = nn.Sequential()
        self.residualNet.add_module('res_conv1', nn.Conv1d(res_inchannel, res_outchannel, kernel_size=1, stride=1))
        self.residualNet.add_module('res_relu1', nn.ReLU(True))
        self.residualNet.add_module('res_conv2', nn.Conv1d(res_outchannel, res_outchannel, kernel_size=1, stride=1))
        self.residualNet.add_module('res_relu2', nn.ReLU(True))
        self.relu = nn.ReLU(True)
        self.Mlp = nn.Sequential(
            nn.Linear(res_outchannel, res_outchannel // 16),
            nn.ReLU(),
            nn.Linear(res_outchannel // 16, res_outchannel),
            nn.Sigmoid()
        )
        self.avg_pool = nn.MaxPool1d(kernel_size=17)
        self.conv = nn.Conv1d(256, 128, kernel_size=1, stride=1)

    def forward(self, x):
        #  需叠加的残差部分
        res_x = self.residualNet(x)
        m = torch.squeeze(self.avg_pool(res_x))
        m = self.Mlp(m)
        scale = m.view(res_x.shape[0], res_x.shape[1], 1)
        res_x = scale * res_x
        residual = self.conv(x)
        out = residual + res_x
        out = self.relu(out)
        return out
        #  (B*128*26)


class NewModel(nn.Module):
    def __init__(self, in_channel, out_channel, res_inchannel, res_outchannel):

        super(NewModel, self).__init__()
        self.channel_attention = ChannelWiseAttention(in_channel)
        self.low_CNN = LowResolutionCNN(in_channel, out_channel)
        self.high_CNN = HighResolutionCNN(in_channel, out_channel)
        self.residual_block = ResidualBlock(res_inchannel, res_outchannel)

    #  class_predictor
        self.class_predictor = nn.Sequential()
        self.class_predictor.add_module('c_fc1', nn.Linear(128 * 26, 100))
        self.class_predictor.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_predictor.add_module('c_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.class_predictor.add_module('c_fc2', nn.Linear(100, 100))
        self.class_predictor.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_predictor.add_module('c_relu2', nn.LeakyReLU(1e-2, inplace=True))
        #self.class_predictor.add_module('c_drop1', nn.Dropout(p=0.3))
        self.class_predictor.add_module('c_fc3', nn.Linear(100, 3))

        self.softmax_c = nn.Softmax(dim=1)

    # domain_discriminator
        self.domain_discriminator0 = nn.Sequential()
        self.domain_discriminator0.add_module('d0_fc1', nn.Linear(128 * 26, 128))
        self.domain_discriminator0.add_module('d0_bn1', nn.BatchNorm1d(128))
        self.domain_discriminator0.add_module('d0_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator0.add_module('d0_fc2', nn.Linear(128, 2))

        self.domain_discriminator1 = nn.Sequential()
        self.domain_discriminator1.add_module('d1_fc1', nn.Linear(128 * 26, 128))
        self.domain_discriminator1.add_module('d1_bn1', nn.BatchNorm1d(128))
        self.domain_discriminator1.add_module('d1_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator1.add_module('d1_fc2', nn.Linear(128, 2))

        self.domain_discriminator2 = nn.Sequential()
        self.domain_discriminator2.add_module('d2_fc1', nn.Linear(128 * 26, 128))
        self.domain_discriminator2.add_module('d2_bn1', nn.BatchNorm1d(128))
        self.domain_discriminator2.add_module('d2_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator2.add_module('d2_fc2', nn.Linear(128, 2))

        self.drop = nn.Dropout(p=0.3)

    def forward(self, data):
        # 输入数据（B*C*L）
        data_c = self.channel_attention(data)
        # print('data_s_c:', data_s_c.shape)
        data_1 = self.low_CNN(data_c)
        # print('data_s1:', data_s1.shape)
        data_2 = self.high_CNN(data_c)
        # print('data_s2:', data_s2.shape)
        data_total = torch.cat((data_1, data_2), dim=2)
        # print('data_s_total:',data_s_total.shape)
        data_total = self.drop(data_total)
        feature = self.residual_block(data_total)
        b, c, l = feature.shape
        feature_s = feature.view(b, c * l)
        # label predictor
        s_label = self.class_predictor(feature_s)
        return s_label