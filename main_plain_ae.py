from training import train_mapper
from models import ConvPlainAE, DeConvPlainAE
from constants import *
from utils import MyDataset
import torch
import torch.optim as optim

batch_size = 128

dataset = 'total'
vae_type = 'conv'

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

# Compose datasets where X are potential samples, y are rays samples. The matching before samples from the same
# class is respected due to the generation of the subsets.
train_dataset = MyDataset(x=potential_train_dataset.data, y=rays_train_dataset.data)
train_dataset.y = train_dataset.y[1, :, :, :].unsqueeze(0)
train_dataset.X = train_dataset.X[1, :, :, :].unsqueeze(0)
test_dataset = MyDataset(x=potential_test_dataset.data, y=rays_test_dataset.data)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

ae = DeConvPlainAE(image_channels=3)

lr = 1e-3
optimizer = optim.Adam(ae.parameters(), lr=lr)

train_mapper(net=ae, train_loader=train_loader, test_loader=test_loader, epochs=600, optimizer=optimizer)
