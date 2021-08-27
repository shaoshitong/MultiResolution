# 生成sample与数据保存,五秒的segment
import scipy.io as sio
import numpy as np

def normalize(data):
    mu = np.expand_dims(np.mean(data,axis=2), axis = 2)
    std = np.expand_dims(np.std(data,axis=2), axis = 2)
    return (data - mu)/std

def sample_DREAMER(data, path):
    t = 5
    # data prepocess
    for i in range(23):
        print("DREAMER person_%d processing" % (i))
        signal_total = np.empty((18, 128 * 60, 14))
        data_p = data['DREAMER'][0, 0]['Data'][0, i]
        data_EEG = data_p[0, 0]['EEG']

        # stimulate(18)
        for j in range(18):
            baseline_signal = data_EEG[0, 0]['baseline'][j, 0]
            signal_part = data_EEG[0, 0]['stimuli'][j, 0]
            signal_part = signal_part[-7680:, :]
            baseline = np.zeros((128, 14))
            for m in range(61):
                baseline = baseline + baseline_signal[128 * m: 128 * (m + 1), :]
            baseline = baseline / 61
            # baseline_expend = baseline
            baseline_expend = np.tile(baseline, (60, 1))
            signal_part = np.expand_dims((signal_part - baseline_expend), axis=0)
            signal_total[j,:,:] = signal_part

        signal_total = signal_total.reshape((-1, 128 * t, 14))
        signal_total = signal_total.transpose(0, 2, 1)

        Score_A = data_p[0, 0]['ScoreArousal']
        Score_V = data_p[0, 0]['ScoreValence']

        Score_A[Score_A < 4] = 0
        Score_A[Score_A >= 4] = 1

        Score_V[Score_V < 4] = 0
        Score_V[Score_V >= 4] = 1

        Score_A = Score_A.reshape(-1, 1)
        Score_A = Score_A.repeat(20, 1)
        Score_A = Score_A.flatten()

        Score_V = Score_V.reshape(-1, 1)
        Score_V = Score_V.repeat(20, 1)
        Score_V = Score_V.flatten()

        # data save
        np.save(path + 'person_%d data' % (i), signal_total)
        np.save(path + 'person_%d label_V' % (i), Score_V)
        np.save(path + 'person_%d label_A' % (i), Score_A)

def sample_SEED(data_path, sample_path):
    t = 5
    label = sio.loadmat(data_path + 'label.mat')
    label = np.array(label['label']) + 1
    label = label.reshape(-1, 1)
    label = label.repeat(int(180 / t), 1)
    label = label.flatten()
    np.save(sample_path + 'label', label)
    person_list = ['djc', 'jl', 'jj', 'lqj', 'ly', 'mhw', 'phl',
                   'sxy', 'wk', 'ww', 'wsf', 'wyw', 'xyl', 'ys', 'zjy']
    for i in range(15):
        print("SEED person_%d processing" % (i))
        signal_total = np.empty([15*3, 62, 36000])
        for j in range(3):
            data = sio.loadmat(data_path + '%d_%d.mat' %(i+1,j+1))
            for k in range(15):
                signal = data[person_list[i] + '_eeg%d' % (k + 1)]
                signal_total[k+j*15,:,:] = signal[:,0:36000]
        signal_total = signal_total.transpose(0, 2, 1)
        signal_total = signal_total.reshape(-1, 200*t, 62)
        signal_total = signal_total.transpose(0, 2, 1)
        signal_total = normalize(signal_total)

        # data save
        np.save(sample_path + 'person_%d' % (i), signal_total)

def feature_SEED(data_path, sample_path):
    label = sio.loadmat(data_path + 'label.mat')
    label = np.array(label['label']) + 1
    label = label.repeat(int(180/20), 1)
    label = label.flatten()
    print(label.shape)
    np.save(sample_path + 'label', label)
    # subjects
    for i in range(15):
        print("SEED person_%d processing" % (i))
        signal_total = np.empty([15*3, 62, 180*5])
        for j in range(3):
            data = sio.loadmat(data_path + '%d_%d.mat' %(i+1,j+1))
            for k in range(15):
                signal = data['de_LDS%d' % (k + 1)]
                signal = signal[:, 0:180,:]
                signal = signal.reshape(62,-1)
                signal_total[k+j*15,:,:] = signal
        signal_total = normalize(signal_total)
        signal_total = signal_total.transpose(0,2,1)
        signal_total = signal_total.reshape(15*3*9, -1, 62)
        signal_total = signal_total.transpose(0, 2, 1)
        # data save
        np.save(sample_path + 'person_%d data' % (i), signal_total)

def sample_DEAP(data_path, sample_path):
    # data prepocess
    t = 5
    person = ['s01', 's02', 's03', 's04', 's05','s06', 's07', 's08', 's09',
              's10', 's11', 's12', 's13','s14', 's15', 's16','s17', 's18',
              's19', 's20', 's21','s22', 's23', 's24','s25', 's26', 's27',
              's28', 's29','s30', 's31', 's32']
    for i in range(32):
        print("DEAP person_%d processing" % (i+1))
        path = data_path + person[i] + '.mat'
        EEG_total = sio.loadmat(path)['data']
        label = sio.loadmat(path)['labels']

        # Score_A = label[:, 1]
        Score_V_ = label[:, 0]

        # 只能用于valence
        # 获取index
        index_V = [k for k in range(40)]

        for j in range(40):
            if 4.8 <= Score_V_[j] <= 5.2:
                index_V.remove(j)

        # 消除无效数据
        EEG = np.ones([len(index_V), EEG_total.shape[1],  EEG_total.shape[2]])
        Score_V = np.ones([len(index_V)])

        for l, k in enumerate(index_V):
            EEG[l,:,:] = EEG_total[k,:,:]
            Score_V[l] = Score_V_[k]

        print(EEG.shape)
        print(Score_V.shape)

        signal_total = EEG[:, 0:32, -7680:]

        baseline = (EEG[:, 0:32, 0:128] + EEG[:, 0:32, 128:128*2] + EEG[:, 0:32, 128 * 2:128*3]) / 3
        baseline = np.tile(baseline, (1,1,60))

        signal_total = (signal_total - baseline).transpose(0,2,1)
        signal_total = signal_total.reshape((-1 , 128*t, 32))
        signal_total = signal_total.transpose(0,2,1)
        signal_total = normalize(signal_total)

        # emotion label
        # Score_A[Score_A <= 5] = 0
        # Score_A[Score_A > 5] = 1

        Score_V[Score_V <= 5] = 0
        Score_V[Score_V > 5] = 1

        # Score_A = Score_A.reshape(-1, 1)
        # Score_A = Score_A.repeat((60/t), 1)
        # Score_A = Score_A.flatten()

        Score_V = Score_V.reshape(-1, 1)
        Score_V = Score_V.repeat(int(60/t), 1)
        Score_V = Score_V.flatten()

        # data save
        np.save(sample_path + 'person_%d data' % (i), signal_total)
        np.save(sample_path + 'person_%d label_V' % (i), Score_V)
        # np.save(sample_path + 'person_%d label_A' % (i), Score_A)

if __name__ == '__main__':

    dataset = 'SEED' # DEAP/DREAMER/SEED

    if dataset == 'DREAMER':
        data_path = 'G:/DREAMER/DREAMER.mat'
        sample_path = 'G:/DREAMER/leave one subject out/'
        data = sio.loadmat(data_path) # 读取数据
        sample_DREAMER(data, sample_path)

    elif dataset == 'DEAP':
        data_path = 'G:/DEAP/DEAP/'
        sample_path = 'G:/DEAP/leave one subject out/'
        sample_DEAP(data_path, sample_path)

    elif dataset == 'SEED':
        data_path = 'F:/SEED/ExtractedFeatures/'
        sample_path = 'F:/SEED/leave one subject out/'
        feature_SEED(data_path, sample_path)


