from torch.utils import data
import torch

# 构建自己的data.dataset类
# 创建__getitem__ __len__ :魔术方法

class Mydataset(data.Dataset):

    def __init__(self, data, label, person):
        self.data = data
        self.label = label
        self.person = person
        # self.transform = transforms.Compose([transforms.ToTensor()])
# 对宽度进行卷积
    def __getitem__(self, index):
        sample = self.data[index,:,:]
        sample = torch.Tensor(sample) # dtype=torch.float32
        label = self.label[index] #.longTensor()
        person = self.person[index]
        return sample, label, person

    def __len__(self):
        return len(self.label)