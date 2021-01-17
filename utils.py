from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
