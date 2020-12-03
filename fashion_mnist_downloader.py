from torchvision import datasets, transforms
from constants import *

# Fashion MNIST Dataset
train_dataset = datasets.FashionMNIST(root=DATA_ROOT+'fashion_mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root=DATA_ROOT+'fashion_mnist_data', train=False, transform=transforms.ToTensor(), download=False)
