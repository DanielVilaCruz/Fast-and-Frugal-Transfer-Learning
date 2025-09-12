import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset, Subset, random_split
import torchvision
from torchvision import datasets, transforms
import ssl
import urllib.request
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import shutil
import random
from pathlib import Path
from collections import defaultdict
from torchvision.transforms import ConvertImageDtype
from datasets import load_dataset
import h5py

def get_data_brain_cancer(data_root='./data/brain_cancer', 
                         train_batch_size=256, 
                         val_batch_size=256,
                         test_batch_size=256, 
                         data_size=224,
                         train_ratio=0.7,
                         val_ratio=0.15,
                         random_seed=42):    
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # transform_train = transforms.Compose([
    #     transforms.Resize((data_size, data_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # ])

    transform_train = transforms.Compose([
        transforms.Resize(data_size),
        transforms.ToTensor(),
        # ConvertImageDtype(torch.float16), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(data_size, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    #     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    # ])
    
    # Basic transform for validation and test
    transform_test = transforms.Compose([
        transforms.Resize(data_size),
        transforms.ToTensor(),
        # ConvertImageDtype(torch.float16), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Load full dataset (will be split)
    full_dataset = ImageFolder(root=data_root, transform=transform_test)
    
    # Get class names from folder structure
    class_names = full_dataset.classes

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    if isinstance(full_dataset, torch.utils.data.Subset):
        full_dataset.dataset.transform = transform_test
        train_dataset.dataset.transform = transform_train
    else:
        train_dataset.dataset.transform = transform_train

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_names
    
class BreakHisDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='binary', magnification='all'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.magnification = magnification
        self.samples = []
        self.class_to_idx = {}

        self._load_dataset()

    def _load_dataset(self):
        class_names = set()

        for class_type in ['benign', 'malignant']:
            class_path = os.path.join(self.root_dir, 'breast', class_type)

            if not os.path.isdir(class_path):
                import pdb
                pdb.set_trace()
                continue

            for subtype_root, dirs, files in os.walk(class_path):
                if self.magnification != 'all':
                    if not subtype_root.endswith(self.magnification):
                        continue

                for file in files:
                    if file.lower().endswith('.png'):
                        filepath = os.path.join(subtype_root, file)
                        if self.mode == 'binary':
                            label = class_type
                        elif self.mode == 'multiclass':
                            label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))
                        else:
                            raise ValueError("mode must be 'binary' or 'multiclass'")
                        self.samples.append((filepath, label))
                        class_names.add(label)

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names))}
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    @property
    def targets(self):
        return [label for (_, label) in self.samples]

def get_data_breakhis(
                      max_dataset_size=None, 
                      data_root='./data/break_his/BreaKHis_v1/BreakHis_v1/histology_slides',
                      train_batch_size=256, 
                      val_batch_size=256,
                      test_batch_size=256,
                      data_size=(256, 256),
                      train_ratio=0.85,
                      val_ratio=0.15,
                      mode='binary',  # 'binary' or 'multiclass'
                      magnification='all',  # 'x40', 'x100', 'x200', 'x400', or 'all'
                      random_seed=42, 
                      quant=None):

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    if quant: 
        transform_train = transforms.Compose([
            transforms.Resize(data_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            ConvertImageDtype(quant), 
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # transforms.Normalize((0.8018, 0.6498, 0.7656), (0.1105, 0.1555, 0.1126)) # x40
        ])
        

        transform_test = transforms.Compose([
            transforms.Resize(data_size),
            transforms.ToTensor(),
            ConvertImageDtype(quant), 
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # transforms.Normalize((0.8018, 0.6498, 0.7656), (0.1105, 0.1555, 0.1126)) # x40
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(data_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # transforms.Normalize((0.8018, 0.6498, 0.7656), (0.1105, 0.1555, 0.1126)) # x40
        ])
        

        transform_test = transforms.Compose([
            transforms.Resize(data_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # transforms.Normalize((0.8018, 0.6498, 0.7656), (0.1105, 0.1555, 0.1126)) # x40
        ])

    full_dataset = BreakHisDataset(
        root_dir=data_root,
        transform=transform_test,
        mode=mode,
        magnification=magnification
    )

    if max_dataset_size is not None and max_dataset_size > 0:
        if len(full_dataset) > max_dataset_size:
            # Create a subset of the dataset
            indices = torch.randperm(len(full_dataset))[:max_dataset_size]
            full_dataset = torch.utils.data.Subset(full_dataset, indices)

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    if isinstance(full_dataset, torch.utils.data.Subset):
        full_dataset.dataset.transform = transform_test
        train_dataset.dataset.transform = transform_train
    else:
        train_dataset.dataset.transform = transform_train

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    class_names = list(full_dataset.class_to_idx.keys()) if not isinstance(full_dataset, torch.utils.data.Subset) else list(full_dataset.dataset.class_to_idx.keys())
    return train_loader, val_loader, test_loader, class_names

class PCamDataset(Dataset):
    def __init__(self, x_file, y_file, transform=None):
        self.x_data = h5py.File(x_file, 'r')['x']
        self.y_data = h5py.File(y_file, 'r')['y']
        self.y_data = torch.from_numpy(self.y_data[:]).squeeze()
        self.transform = transform
        self.class_to_idx = {'normal': 0, 'tumor': 1}  # PCam is binary classification

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        image = self.x_data[index]  
        image = Image.fromarray(image.astype('uint8'))

        if self.transform:
            image = self.transform(image)

        label = self.y_data[index].long()
        return image, label

def get_data_pcam(max_dataset_size=None, data_root='./data/pcam',
                 train_batch_size=256, 
                 test_batch_size=256,
                 data_size=(96, 96),
                 train_ratio=0.85,  # Not used if splits are predefined
                 random_seed=42,
                 quant=None):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(data_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(data_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_x_file = f"{data_root}/camelyonpatch_level_2_split_train_x.h5"
    train_y_file = f"{data_root}/camelyonpatch_level_2_split_train_y.h5"
    val_x_file = f"{data_root}/camelyonpatch_level_2_split_valid_x.h5"
    val_y_file = f"{data_root}/camelyonpatch_level_2_split_valid_y.h5"
    test_x_file = f"{data_root}/camelyonpatch_level_2_split_test_x.h5"
    test_y_file = f"{data_root}/camelyonpatch_level_2_split_test_y.h5"

    train_dataset = PCamDataset(train_x_file, train_y_file, transform=transform_train)
    val_dataset = PCamDataset(val_x_file, val_y_file, transform=transform_test)
    test_dataset = PCamDataset(test_x_file, test_y_file, transform=transform_test)

    original_total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"Original splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    if max_dataset_size is not None and max_dataset_size > 0:
        train_ratio_orig = len(train_dataset) / original_total
        val_ratio_orig = len(val_dataset) / original_total
        test_ratio_orig = len(test_dataset) / original_total

        target_train_size = int(max_dataset_size * train_ratio_orig)
        target_val_size = int(max_dataset_size * val_ratio_orig)
        target_test_size = max_dataset_size - target_train_size - target_val_size  

        torch.manual_seed(random_seed)
        train_indices = torch.randperm(len(train_dataset))[:target_train_size]
        val_indices = torch.randperm(len(val_dataset))[:target_val_size]
        test_indices = torch.randperm(len(test_dataset))[:target_test_size]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    class_names = ['normal', 'tumor']
    return train_loader, val_loader, test_loader, class_names
