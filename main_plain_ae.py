from training import train_ae
from models import ConvPlainAE
from constants import *
import torch
import torch.optim as optim

batch_size = 128

dataset = 'rays'

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    dataset_root = RAYS_ROOT
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    dataset_root = POTENTIAL_ROOT

train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset_root + 'training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset_root + 'test_' + dataset + '.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

vae = ConvPlainAE(image_dim=image_size, image_channels=image_channels)

lr = 1e-3
optimizer = optim.Adam(vae.parameters(), lr=lr)

train_ae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=300, optimizer=optimizer, dataset=dataset)
