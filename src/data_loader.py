import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Use the rna_sequences for creating the DataLoader.

class DataLoader:
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers
    
    def load_data(self):
        self.data = pd.read_csv(self.data_dir)
        pass

    def preprocess_data(self):
        pass

class RNADataset(Dataset):
    def __init__(self, sequences, transform=None):
        self.data = sequences
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_seq = self.sequences[idx]
        pass