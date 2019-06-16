import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

_DATASETS_MAIN_PATH = '/home/ramin/Download/Datasets'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'svhn': os.path.join(_DATASETS_MAIN_PATH, 'SVHN'),
    'emnist': os.path.join(_DATASETS_MAIN_PATH, 'EMNIST'),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    }
}


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'stl10':
        return datasets.CIFAR100(root=_dataset_path['stl10'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.CIFAR100(root=_dataset_path['mnist'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'svhn':
        return datasets.CIFAR100(root=_dataset_path['svhn'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)
    else:
        raise Exception('Ramin: dataset name is wrong/not registered')


def get_num_classes(name):
    if name == 'cifar10':
        return 10
    elif name == 'cifar100':
        return 100
    elif name == 'stl10':
        return 10
    elif name == 'mnist':
        return 10
    elif name == 'svhn':
        return 10
    elif name == 'imagenet':
        return 1000
    else:
        raise Exception('Ramin: dataset name is wrong/not registered')
