from training import train_vae
from models import ConvVAE, ConvVAETest
from constants import *
import torch
import torch.optim as optim

batch_size = 8
vae_type = 'conv'

dataset = 'rays'

train_dataset = torch.load(DATA_ROOT + 'D=0.3 num=999/loaded_data/' + 'training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'D=0.3 num=999/loaded_data/' + 'test_' + dataset + '.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    # hidden_size = RAYS_HIDDEN_SIZE
    hidden_size = 8 * 145 * 145
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

vae = ConvVAETest(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
                  net_size=1)

lr = 1e-4
optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=0)

recon_weight = 1.
kl_weight = 1.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=300, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset, nn_type=vae_type, is_L1=False, exponent=4)
