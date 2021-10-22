from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

"""
数据结构：[45, 62, 900]
用不同长度的kernel提取特征后经过resnet结构对dimension=900进行处理进行multihead 
main3进行训练
"""


# 定义GRL
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# Scaled dot product attention
def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    # dropout设置参数，避免p_atten过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(3)])
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value):
        batches = query.size(0)
        residual = query
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query = query.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        # value = value.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        query, key, value = [l(x).view(batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.(B,5,180,62)
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batches, -1, self.h * self.d_k)
        # linear out + dropout(用新的线性层)(B,180,310)
        x = self.dropout(self.linear(x))
        # Residual connection + layerNorm
        x += residual
        x = self.layer_norm(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, h, d_model):
        super(AttentionLayer, self).__init__()
        print(h,d_model)
        self.self_attn = MultiHeadAttention(h, d_model, dropout=0.5)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        return x

class Feature_Get_Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=5,size=900):
        super(Feature_Get_Block, self).__init__()
        self.inf=nn.Sequential(*[
            nn.Conv1d(in_channels, out_channels, kernel_size=(stride,), stride=(stride,)),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        ])
        if int(size//stride)%3==0:
            k=3
        elif int(size//stride)%2==0:
            k=2
        else:
            k=1
        self.selfattention=AttentionLayer(k,int(size//stride))
    def forward(self,x):
        return self.selfattention(self.inf(x))
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels=62, out_channels=62,size=900,nums_parallel=4):
        super(MultiScaleBlock, self).__init__()
        self.block = nn.ModuleList([
            Feature_Get_Block(in_channels,out_channels,stride=5*(i+1),size=size) for i in range(nums_parallel)
        ])
        self.out_size=sum([int(size//(5*(i+1))) for i in range(nums_parallel)])
    def forward(self, x):
        result=[]
        for layer in self.block:
            result.append(layer(x))
        x_cat = torch.cat(result, dim=2)
        return x_cat
class ResNetBlock(nn.Module):
    def __init__(self,in_planes,planes,stride,dropout=0.1):
        super(ResNetBlock,self).__init__()
        self.res=nn.Sequential(*[
            nn.Conv1d(in_planes,planes,(3,),(stride),(1,),bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(planes),
            nn.Conv1d(planes, planes, (3,), (1,),(1,) ,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(planes),
        ])
        self._initialize()
    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer,nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight.data,mode="fan_in",nonlinearity="relu")
            elif isinstance(layer,nn.BatchNorm1d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.weight.data)
    def forward(self,x):
        return x+self.res(x)
class MultiScaleBlockLayer(nn.Module):
    def __init__(self,in_channels,size=900,nums_layer=1):
        super(MultiScaleBlockLayer,self).__init__()
        self.model=nn.ModuleList([])
        self.turn_model=nn.ModuleList([])
        for i in range(nums_layer):
            A=MultiScaleBlock(in_channels,in_channels,size)
            self.model.append(A)
            self.turn_model.append(ResNetBlock(in_channels,in_channels,1))
        self.out_size=A.out_size*nums_layer

    def forward(self,x):
        p=[]
        for layer,turn in zip(self.model,self.turn_model):
            m=layer(x)
            p.append(m)
            x=turn(x)
        return torch.cat(p,dim=2)
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hid_channels),
            nn.Conv1d(hid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.residual = nn.Sequential()
        if out_channels != in_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        x_residual = self.residual(x)
        x_out = self.block(x)
        out = x_out + x_residual
        return out
'''


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.multiScaleBlock =MultiScaleBlockLayer(62,900,nums_layer=4)
        self.global_attention = AttentionLayer(4, self.multiScaleBlock.out_size)
        self.out_size=self.multiScaleBlock.out_size
    def forward(self, x):
        x_out = self.multiScaleBlock(x)
        x_output = self.global_attention(x_out)
        return x_output


class OverallModel(nn.Module):
    def __init__(self, num_classes=3):
        super(OverallModel, self).__init__()
        self.feature_extraction = FeatureExtraction()
        # bottleneck layer
        self.bottleneck_layer = nn.Sequential(
                                                nn.Linear( 62*self.feature_extraction.out_size, 2048),
                                              nn.ReLU(),
                                              nn.Dropout(),
                                              nn.Linear(2048, 512),
                                              nn.ReLU(),
                                              nn.Dropout()
                                              )
        # class predictor
        self.class_predictor = nn.Sequential(nn.Linear(512, num_classes)
                                             )
        # local domain discriminator
        self.multi_domain = nn.ModuleList([nn.Sequential(
            nn.Linear(512, 2)
        )] * 3)

    def forward(self, data_s, data_t, alpha):
        # feature extraction
        feature_s = self.feature_extraction(data_s)
        feature_s_reshape = feature_s.reshape(data_s.size(0), -1)
        feature_t = self.feature_extraction(data_t)
        feature_t_reshape = feature_t.reshape(data_t.size(0), -1)

        # label predict
        bottle_feature_s_reshape = self.bottleneck_layer(feature_s_reshape)
        s_label_out = self.class_predictor(bottle_feature_s_reshape)
        bottle_feature_t_reshape = self.bottleneck_layer(feature_t_reshape)
        t_label_out = self.class_predictor(bottle_feature_t_reshape)

        # domain predict
        local_domain_s = []
        local_domain_t = []

        if self.training:
            reverse_feature_s = ReverseLayerF.apply(bottle_feature_s_reshape, alpha)
            reverse_feature_t = ReverseLayerF.apply(bottle_feature_t_reshape, alpha)
            for class_index in range(3):
                ps = F.softmax(s_label_out, dim=1)[:, class_index].unsqueeze(1) * reverse_feature_s
                pt = F.softmax(t_label_out, dim=1)[:, class_index].unsqueeze(1) * reverse_feature_t
                local_domain_s.append(
                    self.multi_domain[class_index](ps)
                    )
                local_domain_t.append(
                    self.multi_domain[class_index](pt)
                )
        else:
            pass
        return s_label_out, local_domain_s, local_domain_t, bottle_feature_s_reshape, bottle_feature_t_reshape


'''
# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff, bias=False)
#         self.w_2 = nn.Linear(d_ff, d_model, bias=False)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         residual = x
#         x = self.w_2(self.dropout(F.relu(self.w_1(x))))
#         x += residual
#         x = self.layer_norm(x)
#         return x


# class EncoderLayer(nn.Module):
#     def __init__(self, h, d_model, d_ff):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(h, d_model, dropout=0.5)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=0.5)
#
#     def forward(self, x):
#         x = self.self_attn(x, x, x)
#         x = self.feed_forward(x)
#         return x

# class Encoder(nn.Module):
#     def __init__(self, h, d_model, d_ff):
#         super(Encoder, self).__init__()
#         self.layers = nn.ModuleList([EncoderLayer(h, d_model, d_ff) for _ in range(2)])
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, hid_channels, out_channels):
#         super(ResBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv1d(in_channels, hid_channels, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(hid_channels),
#             nn.Conv1d(hid_channels, out_channels, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(out_channels),
#         )
#         self.residual = nn.Sequential()
#         if out_channels != in_channels:
#             self.residual = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
#                 nn.BatchNorm1d(out_channels)
#             )
#
#     def forward(self, x):
#         x_residual = self.residual(x)
#         x_out = self.block(x)
#         out = x_out + x_residual
#         return out
'''