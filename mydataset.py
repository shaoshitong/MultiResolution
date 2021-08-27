from torch.utils.data import Dataset
import numpy as np
import torch

class Mydataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
        # self.transform = transforms.Compose([transforms.ToTensor()])
        # 对宽度进行卷积

    def __getitem__(self, index):
        sample = self.data[index, :, :]
        sample = torch.Tensor(sample)  # dtype=torch.float32转换为张量类型
        label = self.label[index]  # .longTensor()
        return sample, label

    def __len__(self):#数据集长度
        return len(self.label)