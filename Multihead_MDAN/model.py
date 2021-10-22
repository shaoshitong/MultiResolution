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

class LearnedPositionEmbedding(nn.Embedding):
    def __init__(self,max_len=1000,dropout=0.1,d_model=1):
        super(LearnedPositionEmbedding,self).__init__(d_model,max_len)
        self.max_len=max_len
        self.dropout=dropout
    def forward(self,input):
        weight=self.weight.data
        input=input+weight[:,:input.shape[-1]].unsqueeze(0)
        if input.requires_grad==True:
            input=F.dropout(input,p=self.dropout)
        return input

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
        query = query.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        # query, key, value = [l(x).view(batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
        #                      zip(self.linears, (query, key, value))]
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
        self.self_attn = MultiHeadAttention(h, d_model, dropout=0.5)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        return x


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


class MultiResolutionBlock(nn.Module):
    def __init__(self, in_channels=62, out_channels=62):
        super(MultiResolutionBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.block1.apply(init_weights)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=25, stride=10, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.block2.apply(init_weights)
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=75, stride=15, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.block3.apply(init_weights)
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=150, stride=25, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.block4.apply(init_weights)
        self.bottleneck1 = nn.Linear(900, 512)
        self.bottleneck1.apply(init_weights)
        self.bottleneck2 = nn.Linear(355, 512)
        self.bottleneck2.apply(init_weights)

    def forward(self, x):
        residual = self.bottleneck1(x)
        x1, x2, x3, x4 = self.block1(x), self.block2(x), self.block3(x), self.block4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=2)
        x_out = self.bottleneck2(x_cat)
        x_output = x_out + residual
        return x_output


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        # self.multiResolution = MultiResolutionBlock()
        # self.attention = AttentionLayer(h=8, d_model=512)
        self.inf=nn.Sequential(
                            nn.LayerNorm([62,900]),
                            nn.MaxPool1d((10,),(10,)),
                            nn.Conv1d(62,1024,(1,),(1,),bias=False))
        self.convf=nn.Sequential(
            nn.LayerNorm([62,900]),
                                    nn.LeakyReLU(),
                                    nn.Conv1d(62,2048,(10,),(10,),(0,),bias=False),
                                    nn.LeakyReLU(),
            nn.LayerNorm([2048,90]),
                                    nn.Conv1d(2048,1024,(5,),(1,),(2,),bias=False)
        )
        self.convf.apply(init_weights)
        self.inf.apply(init_weights)
        self.Transformer_Encoder_a=nn.ModuleList([
            nn.TransformerEncoderLayer(1024,8,2048,dropout=0.1,batch_first=True) for _ in range(1)
        ])
        self.Transformer_Decoder_b=nn.ModuleList([
             nn.TransformerDecoderLayer(1024,8,2048,dropout=0.1,batch_first=True) for _ in range(1)
        ])
        self.embedding=LearnedPositionEmbedding()
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, x):
        # 8,62,900
        a=self.convf(x)+self.inf(x)
        a = self.embedding(a)
        """
        a=a.permute(0,2,1)
        m=a
        for encoder in self.Transformer_Encoder_a:
            m=encoder(m)
        for decoder in self.Transformer_Decoder_b:
            a=decoder(a,m)
        # 8,180,512
        a=a.permute(0,2,1)
        """
        return a


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif classname.find('Linear') != -1:
        nn.init.zeros_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)



class OverallModel(nn.Module):
    def __init__(self, num_classes=3):
        super(OverallModel, self).__init__()
        self.feature_extraction = FeatureExtraction()
        # bottleneck layer
        # class predictor
        self.outfn=nn.Sequential(
            nn.LayerNorm([1024,90]),
            nn.ReLU(),
            nn.MaxPool1d(90),
            nn.Flatten()

        )
        self.class_predictor = nn.Sequential(nn.Linear(1024, num_classes)
                                             )
        self.class_predictor.apply(init_weights)
        # local domain discriminator
        self.multi_domain = nn.ModuleList([nn.Sequential(
            nn.Linear(1024, 2)
        )] * 3)
        self.multi_domain.apply(init_weights)

    def forward(self, data_s, data_t, alpha):
        # feature extraction
        feature_s = self.feature_extraction(data_s)
        feature_s_reshape = self.outfn(feature_s)
        feature_t = self.feature_extraction(data_t)
        feature_t_reshape =  self.outfn(feature_t)

        # label predict
        bottle_feature_s_reshape = feature_s_reshape
        s_label_out = self.class_predictor(bottle_feature_s_reshape)
        bottle_feature_t_reshape = feature_t_reshape
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