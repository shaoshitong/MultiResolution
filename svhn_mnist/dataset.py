import torch
from collections import Counter
from getdata import get_dataset,name2benchmark
import sys,os,copy,random
import torchvision
sys.path.append('../')
from RandAugment import RandAugment
import PIL.Image as Image
import utils
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler


class DatasetWithIndicesWrapper(torch.utils.data.Dataset):

    def __init__(self, name, data, targets, transforms, base_transforms):
        self.name = name
        self.data = data
        self.targets = targets
        self.transforms = transforms
        self.base_transforms = base_transforms
        self.rand_aug_transforms = copy.deepcopy(self.base_transforms)
        self.committee_size = 1
        self.ra_obj = RandAugment(1, 2.0)

        if self.name == 'mnist':  # See if this can be dropped entirely without losing performance
            self.transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
            self.base_transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
            self.rand_aug_transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))

        self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        data, target = self.data[index], self.targets[index]

        if self.name == 'mnist':
            data = Image.fromarray(data.numpy(), mode='L')
            data = data.convert('RGB')
        elif self.name == 'svhn':
            data = Image.fromarray(np.transpose(data, (1, 2, 0)))
        else:
            data = utils.default_loader(self.data[index])

        rand_aug_lst = [self.rand_aug_transforms(data) for _ in range(self.committee_size)]
        return (self.transforms(data), self.base_transforms(data), rand_aug_lst), int(target), int(index)


class UDADataset:
    """
    Dataset Class
    """

    def __init__(self, name, LDS_type, is_target=False, img_dir='data', valid_ratio=0.2, batch_size=128):
        self.name = name
        self.LDS_type = LDS_type
        self.is_target = is_target
        self.img_dir = img_dir
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.train_size = None
        self.train_dataset = None
        self.num_classes = None
        self.train_transforms = None
        self.test_transforms = None

    def get_num_classes(self):
        return self.num_classes

    def get_dsets(self):
        """Generates and return train, val, and test datasets

        Returns:
            Train, val, and test datasets.
        """
        data_class = get_dataset(self.name, self.name, self.img_dir, self.LDS_type, self.is_target)
        print(data_class.name, data_class.img_dir)
        self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms = data_class.get_data()

        self.train_dataset = DatasetWithIndicesWrapper(self.name, train_dataset.data, train_dataset.targets,
                                                       self.train_transforms, self.test_transforms)
        self.val_dataset = DatasetWithIndicesWrapper(self.name, val_dataset.data, val_dataset.targets,
                                                     self.test_transforms, self.test_transforms)
        self.test_dataset = DatasetWithIndicesWrapper(self.name, test_dataset.data, test_dataset.targets,
                                                      self.test_transforms, self.test_transforms)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_loaders(self, shuffle=True, num_workers=0, class_balance_train=False):
        """Constructs and returns dataloaders

        Args:
            shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
            num_workers (int, optional): Number of threads. Defaults to 4.
            class_balance_train (bool, optional): Whether to class-balance train data loader. Defaults to False.

        Returns:
            Train, val, test dataloaders, as well as selected indices used for training
        """
        if not self.train_dataset: self.get_dsets()
        num_train = len(self.train_dataset)
        self.train_size = num_train

        benchmark = name2benchmark[self.name]

        if benchmark in ["MNIST", "SVHN"]:
            indices = list(range(num_train))
            split = int(np.floor(self.valid_ratio * num_train))
            if shuffle == True: np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]
            valid_sampler = SubsetRandomSampler(valid_idx)
        elif benchmark in ["DomainNet", "OfficeHome", "VisDA2017"]:
            train_idx = np.arange(len(self.train_dataset))
            valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))
        else:
            raise NotImplementedError

        if class_balance_train:
            self.train_dataset.data = [self.train_dataset.data[idx] for idx in train_idx]
            self.train_dataset.targets = self.train_dataset.targets[train_idx]
            if hasattr(self.train_dataset, 'targets_copy'): self.train_dataset.targets_copy = \
            self.train_dataset.targets_copy[train_idx]

            targets = copy.deepcopy(self.train_dataset.targets)
            if isinstance(targets, torch.Tensor): targets = targets.numpy()
            count_dict = Counter(targets)
            print(dict(count_dict))
            count_dict_full = {lbl: 0 for lbl in range(self.num_classes)}
            for k, v in count_dict.items(): count_dict_full[k] = v

            count_dict_sorted = {k: v for k, v in sorted(count_dict_full.items(), key=lambda item: item[0])}
            class_sample_count = np.array(list(count_dict_sorted.values()))
            class_sample_count = class_sample_count / class_sample_count.max()
            class_sample_count += 1e-8
            """
            每一个约束至【0-1】
            """
            weights = 1 / torch.Tensor(class_sample_count)
            sample_weights = [weights[l] for l in targets]
            sample_weights = torch.DoubleTensor(np.array(sample_weights))
            train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            train_sampler = SubsetRandomSampler(train_idx)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
                                                   batch_size=self.batch_size, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, \
                                                 batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, train_idx
# data=UDADataset(name="svhn",LDS_type="natural",is_target=False,img_dir="./data",valid_ratio=0,batch_size=128)
# data.get_dsets()
# src_train_loader, src_val_loader, src_test_loader, src_train_idx=data.get_loaders()
# for i,((data_s, _, _), label_s, _) in enumerate(src_train_loader):
#     print(data_s.shape,label_s.shape)
#     break
# data_2=UDADataset(name="mnist",LDS_type="natural",is_target=True,img_dir="./data",valid_ratio=0,batch_size=128)
# data_2.get_dsets()
# tgt_train_loader,tgt_val_loader,tgt_test_loader,tgt_train_idx=data_2.get_loaders()
# for i,((_, data_t_og, data_t_raug), label_t, indices_t) in enumerate(tgt_train_loader):
#     print(data_t_og.shape,label_t.shape)
#     break
#