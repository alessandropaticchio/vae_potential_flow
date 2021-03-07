from training import train_vae
# from models import ConvVAE, ConvPlainAE
from models import ConvVAE, DenseVAE
from constants import *
import torch
import torch.optim as optim


vae_type = 'conv'

batch_size = 8

dataset = 'potential'

'''train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/test_' + dataset + '.pt')'''

train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_' + dataset + '.pt')


train_dataset = train_dataset[0, :,:, :].unsqueeze(0)
test_dataset = test_dataset[0, :,:, :].unsqueeze(0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS


if vae_type == 'conv':
    vae = ConvVAE(image_dim=image_size, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE, image_channels=image_channels, net_size=1)
else:
    vae = DenseVAE()

# vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)

lr = 1e-3
optimizer = optim.Adam(vae.parameters(), lr=lr)

recon_weight = 1.
kl_weight = 1.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=300, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset, nn_type=vae_type)

