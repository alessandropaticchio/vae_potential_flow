from torch.utils.data import Dataset
import torchvision
import numpy as np

class EncodedDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
