from utils import MyDataset
from constants import *
import torch

dataset = 'MNIST'

sample_size = 1000
n_targets = 10
sample_class_size = int(sample_size / n_targets)

if dataset == 'MNIST':
    from mnist_downloader import train_dataset, test_dataset
else:
    from fashion_mnist_downloader import train_dataset, test_dataset

X_train = train_dataset.data
y_train = train_dataset.targets

X_test = test_dataset.data
y_test = test_dataset.targets

# Sample initialization
X_train_subset = torch.empty((1, 1, 28, 28))
X_test_subset = torch.empty((1, 1, 28, 28))

y_train_subset = torch.empty((1, 1))
y_test_subset = torch.empty((1, 1))

for target in range(0, n_targets):
    train_indeces = (y_train == target).nonzero().flatten().tolist()[:sample_class_size]
    test_indeces = (y_test == target).nonzero().flatten().tolist()[:sample_class_size]

    X_train_subset = torch.cat((X_train_subset, X_train[train_indeces].unsqueeze(1)), 0)
    X_test_subset = torch.cat((X_test_subset, X_test[test_indeces].unsqueeze(1)), 0)

    y_train_subset = torch.cat((y_train_subset, y_train[train_indeces].unsqueeze(1)), 0)
    y_test_subset = torch.cat((y_test_subset, y_test[test_indeces].unsqueeze(1)), 0)

#  First tensor is meaningless
X_train_subset = X_train_subset[1:]
X_test_subset = X_test_subset[1:]
y_train_subset = y_train_subset[1:]
y_test_subset = y_test_subset[1:]

X_train_subset -= torch.min(X_train_subset)
X_train_subset /= torch.max(X_train_subset)

X_test_subset -= torch.min(X_test_subset)
X_test_subset /= torch.max(X_test_subset)

train_subset = MyDataset(x=X_train_subset, y=y_train_subset)
test_subset = MyDataset(x=X_test_subset, y=y_test_subset)

torch.save(train_subset, DATA_ROOT + '/subsets/' + dataset + '/training.pt')
torch.save(train_subset, DATA_ROOT + '/subsets/' + dataset + '/test.pt')

