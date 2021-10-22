import torch.nn as nn
import torch
from GRL import ReverseLayerF
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


class MultiResolutionCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 设置in_channel=310,out_channel=128
        super(MultiResolutionCNN, self).__init__()
        self.Resolution1 = nn.Sequential()
        self.Resolution1.add_module('R1_conv1', nn.Conv1d(in_channel, out_channel, kernel_size=6, stride=3))
        self.Resolution1.add_module('R1_relu1', nn.ReLU())
        self.Resolution1.add_module('R1_pool1', nn.MaxPool1d(kernel_size=4, stride=2))
        self.Resolution1.add_module('R1_conv2', nn.Conv1d(out_channel, 2*out_channel, kernel_size=2, stride=1))
        self.Resolution1.add_module('R1_relu2', nn.ReLU())

        self.Resolution2 = nn.Sequential()
        self.Resolution2.add_module('R2_conv1', nn.Conv1d(in_channel, out_channel, kernel_size=20, stride=4))
        self.Resolution2.add_module('R2_relu1', nn.ReLU())
        self.Resolution2.add_module('R2_pool1', nn.MaxPool1d(kernel_size=4, stride=2))
        self.Resolution2.add_module('R2_conv2', nn.Conv1d(out_channel, 2 * out_channel, kernel_size=2, stride=1))
        self.Resolution2.add_module('R2_relu2', nn.ReLU())

        self.Resolution3 = nn.Sequential()
        self.Resolution3.add_module('R3_conv1', nn.Conv1d(in_channel, out_channel, kernel_size=60, stride=10))
        self.Resolution3.add_module('R3_relu1', nn.ReLU())
        self.Resolution3.add_module('R3_pool1', nn.MaxPool1d(kernel_size=4, stride=2))
        self.Resolution3.add_module('R3_conv2', nn.Conv1d(out_channel, 2 * out_channel, kernel_size=2, stride=1))
        self.Resolution3.add_module('R3_relu2', nn.ReLU())

    def forward(self, x):
        x1 = self.Resolution1(x)
        x2 = self.Resolution2(x)
        x3 = self.Resolution3(x)
        x_out = torch.cat((x1, x2), dim=2)
        x_out = torch.cat((x_out, x3), dim=2)
        return x_out


class ResidualBlock(nn.Module):
    def __init__(self, res_inchannel, res_outchannel):
        #  输入为B*256*L,res_inchannel=256, res_outchannel=128
        super(ResidualBlock, self).__init__()
        self.residualNet = nn.Sequential()
        self.residualNet.add_module('res_conv1', nn.Conv1d(res_inchannel, res_outchannel, kernel_size=1, stride=1))
        self.residualNet.add_module('res_relu1', nn.ReLU(True))
        self.residualNet.add_module('res_conv2', nn.Conv1d(res_outchannel, res_outchannel, kernel_size=3, stride=1,
                                                           padding=1))
        self.residualNet.add_module('res_relu2', nn.ReLU(True))
        self.residualNet.add_module('res_conv3', nn.Conv1d(res_outchannel, res_outchannel, kernel_size=1, stride=1))
        self.residualNet.add_module('res_relu3', nn.ReLU(True))
        self.relu = nn.ReLU(True)
        self.Mlp = nn.Sequential(
            nn.Linear(res_outchannel, res_outchannel // 16),
            nn.ReLU(),
            nn.Linear(res_outchannel // 16, res_outchannel),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=49)
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


class NewModel(nn.Module):
    def __init__(self, in_channel, out_channel, res_inchannel, res_outchannel):

        super(NewModel, self).__init__()
        self.channel_attention = ChannelWiseAttention(in_channel)
        self.multi_resolution = MultiResolutionCNN(in_channel, out_channel)
        self.residual_block = ResidualBlock(res_inchannel, res_outchannel)

    #  class_predictor
        self.class_predictor = nn.Sequential()
        self.class_predictor.add_module('c_fc1', nn.Linear(128 * 49, 1024))
        self.class_predictor.add_module('c_bn1', nn.BatchNorm1d(1024))
        self.class_predictor.add_module('c_relu', nn.LeakyReLU(1e-2, inplace=True))
        self.class_predictor.add_module('c_drop1', nn.Dropout(p=0.3))
        self.class_predictor.add_module('c_fc2', nn.Linear(1024, 3))
        self.softmax_c = nn.Softmax(dim=1)

    # domain_discriminator(分类)
        self.domain_discriminator0 = nn.Sequential()
        self.domain_discriminator0.add_module('d0_fc1', nn.Linear(128 * 49, 2048))
        self.domain_discriminator0.add_module('d0_bn1', nn.BatchNorm1d(2048))
        self.domain_discriminator0.add_module('d0_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator0.add_module('d0_fc2', nn.Linear(2048, 1024))
        self.domain_discriminator0.add_module('d0_bn2', nn.BatchNorm1d(1024))
        self.domain_discriminator0.add_module('d0_relu2', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator0.add_module('d0_fc3', nn.Linear(1024, 2))

        self.domain_discriminator1 = nn.Sequential()
        self.domain_discriminator1.add_module('d1_fc1', nn.Linear(128 * 49, 2048))
        self.domain_discriminator1.add_module('d1_bn1', nn.BatchNorm1d(2048))
        self.domain_discriminator1.add_module('d1_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator1.add_module('d1_fc2', nn.Linear(2048, 1024))
        self.domain_discriminator1.add_module('d1_bn2', nn.BatchNorm1d(1024))
        self.domain_discriminator1.add_module('d1_relu2', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator1.add_module('d1_fc3', nn.Linear(1024, 2))

        self.domain_discriminator2 = nn.Sequential()
        self.domain_discriminator2.add_module('d2_fc1', nn.Linear(128 * 49, 2048))
        self.domain_discriminator2.add_module('d2_bn1', nn.BatchNorm1d(2048))
        self.domain_discriminator2.add_module('d2_relu1', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator2.add_module('d2_fc2', nn.Linear(2048, 1024))
        self.domain_discriminator2.add_module('d2_bn2', nn.BatchNorm1d(1024))
        self.domain_discriminator2.add_module('d2_relu2', nn.LeakyReLU(1e-2, inplace=True))
        self.domain_discriminator2.add_module('d2_fc3', nn.Linear(1024, 2))
        self.drop = nn.Dropout(p=0.3)

    def forward(self, mode, data_s, data_t, alpha):
        # 输入数据（B*C*L）
        data_s_c = self.channel_attention(data_s)
        data_s = self.multi_resolution(data_s_c)
        data_s_total = self.drop(data_s)
        feature_s = self.residual_block(data_s_total)
        b_s, c_s, l_s = feature_s.shape
        feature_s = feature_s.view(b_s, c_s * l_s)
        reverse_feature_s = ReverseLayerF.apply(feature_s, alpha)

        data_t_c = self.channel_attention(data_t)
        data_t = self.multi_resolution(data_t_c)
        data_t_total = self.drop(data_t)
        feature_t = self.residual_block(data_t_total)
        b_t, c_t, l_t = feature_t.shape
        feature_t = feature_t.view(b_t, c_t * l_t)
        reverse_feature_t = ReverseLayerF.apply(feature_t, alpha)

        # label predictor
        s_label = self.class_predictor(feature_s)
        # pseudo_s_label = self.softmax_c(s_label)
        t_label = self.class_predictor(feature_t)
        # pseudo_t_label = self.softmax_c(t_label)
        #  class-wised domain adaptation
        domain_s = []
        domain_t = []
        if mode == 'train':
            # 维度转换变成可以用来进行sub-domain分类的维度
            ps0 = s_label[:, 0].reshape(data_s.shape[0], 1)
            fs0 = ps0 * reverse_feature_s
            pt0 = t_label[:, 0].reshape(data_t.shape[0], 1)
            ft0 = pt0 * reverse_feature_t
            out_s0 = self.domain_discriminator0(fs0)
            domain_s.append(out_s0)
            out_t0 = self.domain_discriminator0(ft0)
            domain_t.append(out_t0)

            ps1 = s_label[:, 1].reshape(data_s.shape[0], 1)
            fs1 = ps1 * reverse_feature_s
            pt1 = t_label[:, 1].reshape(data_t.shape[0], 1)
            ft1 = pt1 * reverse_feature_t
            out_s1 = self.domain_discriminator1(fs1)
            domain_s.append(out_s1)
            out_t1 = self.domain_discriminator1(ft1)
            domain_t.append(out_t1)

            ps2 = s_label[:, 2].reshape(data_s.shape[0], 1)
            fs2 = ps2 * reverse_feature_s
            pt2 = t_label[:, 2].reshape(data_t.shape[0], 1)
            ft2 = pt2 * reverse_feature_t
            out_s2 = self.domain_discriminator2(fs2)
            domain_s.append(out_s2)
            out_t2 = self.domain_discriminator2(ft2)
            domain_t.append(out_t2)
        else:
            pass
        return s_label, domain_s, domain_t