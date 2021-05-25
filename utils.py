import numpy as np
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
    indices = []
    for i, d in enumerate(strengths_data):
        if d in picked_strenghts:
            indices.append(i)
    return pics_data[indices], strengths_data[indices]


# KL Annealing
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.full(n_epoch, stop)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.full(n_epoch, stop)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


if __name__ == '__main__':
    L = frange_cycle_linear(start=0.0, stop=4.0, n_epoch=200, n_cycle=1, ratio=0.5)
    import matplotlib.pyplot as plt

    plt.plot(range(len(L)), L)
    plt.show()
    pass
