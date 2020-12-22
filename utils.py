from torch.utils.data import Dataset


class EncodedDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PicsDataset(Dataset):
    def __init__(self, x):
        self.X = x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]