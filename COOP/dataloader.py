from torch.utils import data
from torch.utils.data import random_split
from torchvision import datasets
import os
import py_vars

def get_data(dataset_name, batch_size, transform, shuffle=True, train_size=0.8, val_size=0.1):
    """
    Loads the dataset and splits it into training, validation and test sets. Available datsets:
    ["cifar10", "cifar100", "imagenet_v2", "imagenet_a"]
    :param dataset_name: str: name of the dataset
    :param batch_size: int: batch size
    :param transform: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    :param train_size: float: proportion of the dataset to include in the training set
    :param val_size: float: proportion of the dataset to include in the validation set
    :return: tuple: training, validation and test dataloaders
    """
    if dataset_name == "cifar10":
        download = not (os.path.exists(os.path.join("data/cifar-10-python")))
        dataset = datasets.CIFAR10(root="./data", download=download, transform=transform)
        id2class = {dataset.class_to_idx[c] : c for c in dataset.classes}
    elif dataset_name == "cifar100":
        download = not (os.path.exists(os.path.join("data/cifar-100-python")))
        dataset = datasets.CIFAR100(root="./data", download=download, transform=transform)
        id2class = {dataset.class_to_idx[c] : c for c in dataset.classes}
    elif dataset_name == "imagenet_v2":
        dataset = datasets.ImageFolder(root="./data/imagenetv2-matched-frequency-format-val", transform=transform)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
        id2class = {dataset.class_to_idx[c] : py_vars.num2class_v2[int(c)] for c in dataset.classes}
    elif dataset_name == "imagenet_a":
        dataset = datasets.ImageFolder(root="./data/imagenet-a", transform=transform)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
        id2class = {dataset.class_to_idx[c] : py_vars.num2class[c] for c in dataset.classes}
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    n = len(dataset)
    n_train = int(train_size * n)
    n_val = int(val_size * n)
    n_test = n - n_train - n_val
    
    if(n_train + n_val == 0):
        train_loader, val_loader = None, None
        test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader, list(id2class.values()), id2class