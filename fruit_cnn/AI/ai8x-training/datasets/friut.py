import torch.nn as nn
import torchvision
from torchvision import transforms

import ai8x


def fruit_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data

    image_size = (64, 64)               #图片大小为64X64

    if load_train:
        train_data_path = data_dir + '/fruit/train_datasets/'
        train_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(),      #随机翻转，水平方向
                            transforms.Resize(image_size),
                            transforms.ToTensor(),                  #转化成Tensor
                            # transforms.Normalize(mean=[0.485,0.456,0.406],
                            #                     std=[0.229,0.224,0.225])]),
                            ai8x.normalize(args=args)
                        ])

        train_dataset = torchvision.datasets.ImageFolder(root = train_data_path,
                                             transform = train_transforms)

    else:
        train_dataset = None

    if load_test:
        test_data_path = data_dir + '/fruit/test_datasets/'

        test_transforms = transforms.Compose([
                            transforms.Resize(image_size),  
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.485,0.456,0.406],
                            #                         std=[0.229,0.224,0.225])]) #标准化处理
                            ai8x.normalize(args=args)
                        ])

        test_dataset = torchvision.datasets.ImageFolder(root = test_data_path,
                                             transform = test_transforms)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'fruit',
        'input': (3, 64, 64),
        'output': ('apple', 'banana', 'grape', 'kiwi', 'mango', 'orange', 'pear',
                   'pineapple', 'pomegranate', "strawberry", "watermelon"),
        'loader': fruit_get_datasets,
    },
]
