import scipy.io as sio
import numpy as np
from einops import rearrange
"""
对维度为dim=180做标准化,将不同频率段的DE特征交替分布,每一个通道对应5个频带数据
"""


def normalize(data):  # data: 62*180
    mu = np.expand_dims(np.mean(data, axis=1), axis=1)
    std = np.expand_dims(np.std(data, axis=1), axis=1)
    return (data - mu)/std


def sample_seed(data_path, sample_path):
    # label为字典形式
    label = sio.loadmat(data_path + 'label.mat')
    # 给数据贴标签-0，1，2
    label = np.array(label['label']) + 1
    label = np.tile(label, 3)
    label = label.transpose(1, 0)
    label = label.flatten()
    np.save(sample_path + 'label', label)

    for i in range(15):  # subject
        print("SEED person_%d processing" % (i))
        signal_total = np.empty([15*3, 62, 180, 5])
        for j in range(3):  # session
            data = sio.loadmat(data_path + '%d_%d.mat' %(i+1, j+1))
            for k in range(15):  # film clips
                data_sample = data['de_movingAve%d' % (k + 1)]
                # 62*180*5
                sample = data_sample[:, 0:180, :]
                normal_sample = normalize(sample)
                #  45*62*180*5
                signal_total[k + j * 15, :, :, :] = normal_sample
        signal_total = rearrange(signal_total, 'b c h w -> b c (h w)')
        print(signal_total.shape)
        np.save(sample_path + 'person_%d data' % i, signal_total)


if __name__ == '__main__':
    data_path = 'F:/SEEDdataset/ExtractedFeatures/'
    sample_path = 'G:/Alex/SEED_experiment/leave one subject out/segment7/'
    sample_seed(data_path, sample_path)