from models import ConvVAE
from training import train_vae
from constants import *
from utils import PicsDataset
import torch
import torch.optim as optim

batch_size = 32

dataset = 'potential'
train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/test_' + dataset + '.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    hidden_size = RAYS_HIDDEN_SIZE
    latent_size = int(hidden_size / 2)
    image_channels = RAYS_IMAGE_CHANNELS
    rays_train_set = PicsDataset(train_dataset)
    rays_test_set = PicsDataset(test_dataset)
else:
    image_size = POTENTIAL_IMAGE_SIZE
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = int(hidden_size / 2)
    image_channels = POTENTIAL_IMAGE_CHANNELS
    potential_train_set = PicsDataset(train_dataset)
    potential_test_set = PicsDataset(test_dataset)

vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)

lr = 1e-3
optimizer = optim.Adam(vae.parameters(), lr=lr)

if torch.cuda.is_available():
    vae.cuda()

recon_weight = 1.
kl_weight = 1.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=100, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset)
