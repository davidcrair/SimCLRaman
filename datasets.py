"""Dataset classes and augmentation functions for spectral data.
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# create a custom dataset class
class SpectralDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

# data augmentation for spectral data
class SpectralAugmentation:
    def __init__(self, noise_level_mean=0.04, noise_level_std=0.02, shift_range=5, scaling_range=(0.9, 1.1)):
        self.noise_level_mean = noise_level_mean
        self.noise_level_std = noise_level_std
        self.shift_range = shift_range
        self.scaling_range = scaling_range
    
    def add_noise(self, spectrum):
        # add gaussian noise to the spectrum
        noise = torch.randn_like(spectrum) * np.random.normal(self.noise_level_mean, self.noise_level_std)
        return spectrum + noise
    
    def shift(self, spectrum):
        # shift the spectrum by a random amount to the left or right
        device = spectrum.device
        shift_amount = torch.randint(-self.shift_range, self.shift_range + 1, (1,), device=device)
        if shift_amount > 0:
            return torch.cat([torch.zeros(shift_amount, device=device), spectrum[:-shift_amount]])
        elif shift_amount < 0:
            return torch.cat([spectrum[-shift_amount:], torch.zeros(-shift_amount, device=device)])
        return spectrum
    
    def scale(self, spectrum):
        # scale the intensity of the spectrum by a random factor
        device= spectrum.device
        scaling_factor = torch.empty(1, device=device).uniform_(*self.scaling_range)
        return spectrum * scaling_factor
    
    def __call__(self, spectrum):
        # apply augmentations with some prob.
        augmented = spectrum.clone()
        if torch.rand(1) < 0.7:  # 70% chance of adding noise
            augmented = self.add_noise(augmented)
        if torch.rand(1) < 0.5:  # 50% chance of shifting
            augmented = self.shift(augmented)
        if torch.rand(1) < 0.7:  # 70% chance of scaling
            augmented = self.scale(augmented)
        return augmented

# wrapper for the two augmented views
class SpectralTwoView:
    def __init__(self):
        self.augmentation = SpectralAugmentation()
    
    def __call__(self, x):
        # create two augmented views of the same spectrum
        aug1 = self.augmentation(x)
        aug2 = self.augmentation(x)
        return aug1, aug2


# wrapper for the contrastive dataset
class ContrastiveSpectralDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx][0] # only get the features
        if self.transform:
            return self.transform(x), self.dataset[idx][1]
        return x, self.dataset[idx][1]
    
class AugmentedSpectralDataset(Dataset):
    def __init__(self, features, labels, augment_prob=0.7):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.augment_prob = augment_prob
        self.augmentation = SpectralAugmentation()
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        
        if torch.rand(1) < self.augment_prob:
            x = self.augmentation(x)
            
        if self.labels is not None:
            return x, self.labels[idx]
        return x