from models import ConvVAE, DeConvVAE
from training import train_vae
from constants import *
import torch
import torch.optim as optim

batch_size = 8

dataset = 'rays'
train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/test_' + dataset + '.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    hidden_size = RAYS_HIDDEN_SIZE
    latent_size = int(hidden_size / 2)
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = int(hidden_size / 2)
    image_channels = POTENTIAL_IMAGE_CHANNELS

vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)

lr = 1e-3
optimizer = optim.SGD(vae.parameters(), lr=lr)

recon_weight = 1.
kl_weight = 0.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=800, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset)
