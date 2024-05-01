import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import py_vars

def load_imagenet_A(path:str, batch_size:int, preprocess:transforms.Compose, shuffle:bool=True):
    """
    Loads the images from the ImageNet-A dataset
    :param path: str: path to the ImageNet-A dataset
    :param batch_size: int: batch size
    :param preprocess: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    """
    imagenet_A = datasets.ImageFolder(root=path, transform=preprocess)
    imagenet_A_loader = torch.utils.data.DataLoader(imagenet_A, batch_size=batch_size, shuffle=shuffle)

    id2class = {imagenet_A.class_to_idx[c] : py_vars.num2class[c] for c in imagenet_A.classes}
    return imagenet_A_loader, id2class

def load_imagenet_v2(path:str, batch_size:int, preprocess:transforms.Compose, shuffle:bool=True):
    """
    Loads the images from the ImageNet-V2 dataset
    :param path: str: path to the ImageNet-V2 dataset
    :param batch_size: int: batch size
    :param preprocess: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    """
    imagenet_v2 = datasets.ImageFolder(root=path, transform=preprocess)
    imagenet_v2.classes = sorted(imagenet_v2.classes, key=int)
    imagenet_v2.class_to_idx = {cls: i for i, cls in enumerate(imagenet_v2.classes)}
    
    imagenet_v2_loader = torch.utils.data.DataLoader(imagenet_v2, batch_size=batch_size, shuffle=shuffle)
    
    id2class = {imagenet_v2.class_to_idx[c] : py_vars.num2class_v2[int(c)] for c in imagenet_v2.classes}
    return imagenet_v2_loader, id2class

def load_cifar100(path:str, batch_size:int, preprocess:transforms.Compose, shuffle:bool=True):
    """
    Loads the images from the CIFAR-100 dataset
    :param path: str: path to the CIFAR-100 dataset
    :param batch_size: int: batch size
    :param preprocess: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    """
    cifar100 = datasets.CIFAR100(root=path, download=True, transform=preprocess)
    cifar100_loader = torch.utils.data.DataLoader(cifar100, batch_size=batch_size, shuffle=shuffle)

    id2class = {cifar100.class_to_idx[c] : c for c in cifar100.classes}
    return cifar100_loader, id2class