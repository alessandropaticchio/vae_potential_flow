from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class StrengthDataset(Dataset):
    def __init__(self, x, d):
        self.X = x
        self.D = d

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.D[idx])


class MappingDataset(Dataset):
    def __init__(self, x, y, d):
        self.X = x
        self.y = y
        self.D = d

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx], self.D[idx])


def generate_dataset_from_strength(pics_data, strengths_data, picked_strenghts):
    indeces = []
    for i, d in enumerate(strengths_data):
        if d in picked_strenghts:
            indeces.append(i)
    return pics_data[indeces], strengths_data[indeces]
