from torchvision import datasets, transforms
from constants import *

# MNIST Dataset
train_dataset = datasets.MNIST(root=DATA_ROOT+'mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=DATA_ROOT+'mnist_data', train=False, transform=transforms.ToTensor(), download=False)
