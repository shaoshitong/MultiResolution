# 生成sample与数据保存,五秒的segment
import scipy.io as sio
import numpy as np
import os


def normalize(data):  # data: 45*310*180
    mu = np.expand_dims(np.mean(data, axis=2), axis=2)
    std = np.expand_dims(np.std(data, axis=2), axis=2)
    return (data - mu)/std

def sample_DREAMER(data, path):
    t = 5#设置样本切分时间为5秒
    # data prepocess
    for i in range(23):#人数为23人，从0开始计数到22
        print("DREAMER person_%d processing" % (i))
        signal_total = np.empty((1, 128*60, 14))
        #读取数据
        data_p = data['DREAMER'][0, 0]['Data'][0, i]#读取第i个人的数据
        data_EEG = data_p[0, 0]['EEG']
        Score_A = data_p[0, 0]['ScoreArousal']
        Score_V = data_p[0, 0]['ScoreValence']
        # stimulate(18)
        for j in range(18):
            baseline_signal = data_EEG[0, 0]['baseline'][j, 0]
            signal_part = data_EEG[0, 0]['stimuli'][j, 0]#第j个视频刺激数据
            signal_part = signal_part[-7680:, :]#读取第j个视频刺激最后60秒数据
            baseline = np.zeros((128,14))
            for m in range(61):
                baseline = baseline + baseline_signal[128 * m: 128 * (m+1), :]
            baseline = baseline / 61#每秒基线数据128*14
            # baseline_expend = baseline
            baseline_expend = np.tile(baseline, (60, 1))#扩充到7680*14（128*14整体复制60次）
            signal_part =  np.expand_dims((signal_part - baseline_expend), axis=0)#shape从（76080,14）转换为（1,7680,14），维度扩展函数
            signal_total = np.append(signal_total, signal_part, axis=0)#扩展为19*7680*14
        signal_total = signal_total[1:, :, :]#获取18*7680*14的数据
        signal_total = signal_total.reshape((-1, 128 * t, 14))#(18*12)*(128*t)*14
        signal_total = signal_total.transpose(0, 2, 1)#转换为（18*12)*14*(128*t)
        #signal_total = np.expand_dims(signal_total,axis=1)#提升维度（18*12)*1*14*(128*t)

        Score_A[Score_A < 4] = 0
        Score_A[Score_A >= 4] = 1

        Score_V[Score_V < 4] = 0
        Score_V[Score_V >= 4] = 1

        Score_A = Score_A.reshape(-1, 1)#给18个刺激打标签
        Score_A = Score_A.repeat(int(60/t), 1)#给同一个刺激不同segment打上相同的情感标签，重复12次
        Score_A = Score_A.flatten()

        Score_V = Score_V.reshape(-1, 1)
        Score_V = Score_V.repeat(int(60/t), 1)
        Score_V = Score_V.flatten()

        # data save
        np.save(path + 'person_%d data' % (i), signal_total)
        np.save(path + 'person_%d label_V' % (i), Score_V)
        np.save(path + 'person_%d label_A' % (i), Score_A)

def sample_DEAP(data_path, sample_path):
    # data prepocess
    t = 5
    for i, path in enumerate(os.listdir(data_path)):#生成列表字典['s01.mat','s02.mat',...]
        print("DEAP person_%d processing" % (i))
        EEG = sio.loadmat(data_path+path)['data']
        label = sio.loadmat(data_path+path)['labels']
        Score_V = label[:, 0]
        Score_A = label[:, 1]
        signal_total = EEG[:, 0:32, -7680:]
        baseline = (EEG[:, 0:32, 0:128] + EEG[:, 0:32, 128:128 * 2] + EEG[:, 0:32, 128 * 2:128 * 3]) / 3
        #40*32*128
        baseline = np.tile(baseline, (1,1,60))
        signal_total = (signal_total - baseline).transpose(0,2,1)#40*7680*32
        signal_total = signal_total.reshape((-1 , 128*t, 32))#(40*12)*640*32
        signal_total = signal_total.transpose(0,2,1)
        #signal_total= np.expand_dims(signal_total,axis=1)

        # emotion label
        Score_A[Score_A <= 5] = 0
        Score_A[Score_A > 5] = 1
        Score_V[Score_V <= 5] = 0
        Score_V[Score_V > 5] = 1
        Score_A = Score_A.reshape(-1, 1)
        Score_A = Score_A.repeat(int(60/t), 1)#40*12，将40*1的标签按列复制12份，40*12
        Score_A = Score_A.flatten()#打平，生成一维数组，480*1
        Score_V = Score_V.reshape(-1, 1)#生成二维数组
        Score_V = Score_V.repeat(int(60/t), 1)
        Score_V = Score_V.flatten()
        # data save
        np.save(sample_path + 'person_%d data' % (i), signal_total)
        np.save(sample_path + 'person_%d label_V' % (i), Score_V)
        np.save(sample_path + 'person_%d label_A' % (i), Score_A)


def sample_seed(data_path, sample_path):
    # label为字典形式
    label = sio.loadmat(data_path + 'label.mat')
    # 标签数据0，1，2
    label = np.array(label['label']) + 1
    label = np.tile(label,3)
    label = label.transpose(1,0)
    label = label.flatten()
    np.save(sample_path + 'label', label)
    for i in range(15):  # subject
        print("SEED person_%d processing" % (i))
        signal_total = np.empty([15*3, 62*5, 180])
        for j in range(3):  # session
            data = sio.loadmat(data_path + '%d_%d.mat' %(i+1,j+1))
            for k in range(15):  # film clips
                signal = data['de_LDS%d' % (k + 1)]
                signal1 = signal[:, 0:180, 0]
                signal2 = signal[:, 0:180, 1]
                signal_cat = np.concatenate((signal1, signal2),axis=0)
                for m in range(3):
                    signal3 = signal[:, 0:180, m+2]
                    signal_cat = np.concatenate((signal_cat, signal3), axis=0)  # 310*180
                signal_total[k+j*15,:,:] = signal_cat

        signal_total = normalize(signal_total)
        print(signal_total.shape)
        # data save
        np.save(sample_path + 'person_%d data' % (i), signal_total)


if __name__ == '__main__':
    dataset = 'SEED' # DEAP
    if dataset == 'DREAMER':
        data_path = 'F:/fc/dreamer_EEG/DREAMER.mat'
        sample_path = 'F:/Alex/DREAMER/leave one subject out/segment1/'
        data = sio.loadmat(data_path) # 读取数据
        sample_DREAMER(data, sample_path)
    elif dataset == 'DEAP':
        data_path = 'F:/fc2/deep_eeg_psd/data_preprocessed_matlab/'
        sample_path = 'F:/Alex/DEAP/leave one subject out/segment1/'
        sample_DEAP(data_path, sample_path)
    elif dataset == 'SEED':
        data_path = 'D:/SEED/ExtractedFeatures/'
        sample_path = 'F:/Alex/SEED/leave one subject out/segment/'
        sample_seed(data_path, sample_path)
