"""Copyright (c) Hyperconnect, Inc. and its affiliates.
All rights reserved.

Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100


from data.MySampler import *

# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default', resolution=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        # get image number list
        occur_dict = dict(Counter(self.labels))
        self.img_num_list = [occur_dict[i] for i in sorted(occur_dict.keys())]

        # select top k class
        if top_k:
            # only select top k in training, in case train/val/test not matching.
            if 'train' in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key = lambda x:x[1], reverse=True)
                # saving
                torch.save(dist, template + '_top_{}_mapping'.format(top_k))
            else:
                # loading
                dist = torch.load(template + '_top_{}_mapping'.format(top_k))
            selected_labels = {item[0]:i for i, item in enumerate(dist[:top_k])}
            # replace original path and labels
            self.new_img_path = []
            self.new_labels = []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    self.new_img_path.append(path)
                    self.new_labels.append(selected_labels[label])
            self.img_path = self.new_img_path
            self.labels = self.new_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

# Load datasets
def load_data(data_root, dataset, phase, batch_size, batch_size_tst_val, top_k_class=None,
              num_workers=4, shuffle=True, cifar_imb_ratio=None,
              test_imb_ratio=None, reverse=False, resolution=224):

    txt_split = phase
    if dataset == "Places_LT":
        txt = f"./data/Places_LT_v2/Places_LT_{phase}.txt"
        template = None
    else:
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
        template = './data/%s/%s'%(dataset, dataset)

    print('Loading data from %s' % (txt))

    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'

    if dataset == 'CIFAR10_LT':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root,
                                test_imb_ratio=test_imb_ratio, reverse=reverse, resolution=resolution)
    elif dataset == 'CIFAR100_LT':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root,
                                 test_imb_ratio=test_imb_ratio, reverse=reverse, resolution=resolution)
    else:
        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key, resolution)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key, resolution)
        print('Use data transformation:', transform)

        set_ = LT_Dataset(data_root, txt, transform, template=template, top_k=top_k_class)


    print(len(set_))

    # count imblanced # of samples per-class

    num_classes = len(np.unique(set_.labels))
    imb_num_per_cls = np.zeros(num_classes) # ordered from class0 to classN
    for idx in range(len(set_.labels)):
        lb = set_.labels[idx]
        imb_num_per_cls[lb] += 1
    # np.save(str(dataset)+'_'+str(phase)+'_imb_num_per_cls.npy', imb_num_per_cls)


    if phase =='train':
        print('=====> Shuffle is %s.' % (shuffle))
        print(f'=====> Phase {phase} loader created')
        print(f"{phase} loader # of sample per-class:\n {imb_num_per_cls}")
        return torch.FloatTensor(set_.img_num_list) / torch.FloatTensor(set_.img_num_list).sum(), \
                DataLoader(dataset=set_,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           ), \
                imb_num_per_cls
    else:
        print('=====> No sampler.')
        print(f'=====> Phase {phase} loader created')
        print(f"{phase} loader # of sample per-class:\n {imb_num_per_cls}")
        return torch.FloatTensor(set_.img_num_list) / torch.FloatTensor(set_.img_num_list).sum(), \
               DataLoader(dataset=set_,
                          batch_size=batch_size_tst_val,
                          shuffle=False,
                          num_workers=num_workers,
                          ), \
               imb_num_per_cls

