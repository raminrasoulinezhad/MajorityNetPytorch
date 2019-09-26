import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#_DATASETS_MAIN_PATH = '/home/ramin/Download/Datasets'
_DATASETS_MAIN_PATH = '../Datasets'
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

    if name == 'cifar10':
        split = (split == 'train')
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=split,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        split = (split == 'train')
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=split,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'stl10':
        if (split == 'val'):
            split = 'test' 
        return datasets.STL10(root=_dataset_path['stl10'],
                                 split=split,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        split = (split == 'train')
        return datasets.MNIST(root=_dataset_path['mnist'],
                                 train=split,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'svhn':
        if (split == 'val'):
            split = 'test' 
        return datasets.SVHN(root=_dataset_path['svhn'],
                                 split=split,
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
