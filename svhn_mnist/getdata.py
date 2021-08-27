# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import numpy as np
import torch
datasets = {}
from torchvision import datasets as dataset_1, transforms
name2benchmark = {
                    "mnist": "MNIST", "svhn": "SVHN"
			    }


def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def get_dataset(name, *args):
    if name not in name2benchmark: return None
    return datasets[name2benchmark[name]](*args)

@register_dataset('MNIST')
class MNISTDataset:
	"""
	MNIST Dataset class
	"""

	def __init__(self, name, img_dir, LDS_type, is_target):
		self.name = name
		self.img_dir = img_dir
		self.LDS_type = LDS_type
		self.is_target = is_target

	def get_data(self):
		mean, std = 0.5, 0.5
		normalize_transform = transforms.Normalize((mean,), (std,))
		self.train_transforms = transforms.Compose([
								   transforms.ToTensor(),
								   normalize_transform
							   ])
		self.test_transforms = transforms.Compose([
								   transforms.ToTensor(),
								   normalize_transform
								])

		self.train_dataset = dataset_1.MNIST(self.img_dir, train=True, download=True)
		self.val_dataset = dataset_1.MNIST(self.img_dir, train=True, download=True)
		self.test_dataset = dataset_1.MNIST(self.img_dir, train=False, download=True)
		self.train_dataset.name, self.val_dataset.name, self.test_dataset.name = 'DIGITS','DIGITS', 'DIGITS'
		self.num_classes = 10
		return self.num_classes, self.train_dataset, self.val_dataset, self.test_dataset, self.train_transforms, self.test_transforms

@register_dataset('SVHN')
class SVHNDataset:
	"""
	SVHN Dataset class

	"""
	def __init__(self, name, img_dir, LDS_type, is_target):
		self.name = name
		self.img_dir = img_dir
		self.LDS_type = LDS_type
		self.is_target = is_target

	def get_data(self):
		mean, std = 0.5, 0.5
		normalize_transform = transforms.Normalize((mean,), (std,))
		RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))
		self.train_transforms = transforms.Compose([
							   RGB2Gray,
							   transforms.Resize((28, 28)),
							   transforms.ToTensor(),
							   normalize_transform
						   ])
		self.test_transforms = transforms.Compose([
							   RGB2Gray,
							   transforms.Resize((28, 28)),
							   transforms.ToTensor(),
							   normalize_transform
						   ])

		self.train_dataset = dataset_1.SVHN(self.img_dir, split='train', download=True)
		self.val_dataset = dataset_1.SVHN(self.img_dir, split='train', download=True)
		self.test_dataset = dataset_1.SVHN(self.img_dir, split='test', download=True)
		self.train_dataset.targets, self.val_dataset.targets, self.test_dataset.targets = \
										self.train_dataset.labels, self.val_dataset.labels, self.test_dataset.labels
		self.num_classes = 10
		return self.num_classes, self.train_dataset, self.val_dataset, self.test_dataset, self.train_transforms, self.test_transforms
