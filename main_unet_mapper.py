from models import *
from training import train_unet_vae
from constants import *
from utils import MyDataset
import torch
import torch.optim as optim

dataset = 'total'
vae_type = 'conv'

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

# Compose datasets where X are potential samples, y are rays samples. The matching before samples from the same
# class is respected due to the generation of the subsets.
train_dataset = MyDataset(x=potential_train_dataset.data, y=rays_train_dataset.data)
test_dataset = MyDataset(x=potential_test_dataset.data, y=rays_test_dataset.data)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

hidden_size = HIDDEN_SIZE
latent_size = LATENT_SIZE
vae = UNet_VAE(hidden_size=HIDDEN_SIZE, latent_size=latent_size)

optimizer = optim.Adam(vae.parameters(), weight_decay=0.)

if torch.cuda.is_available():
    vae.cuda()

recon_weight = 1.
kl_weight = 1.

train_unet_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer,
               recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset, nn_type=vae_type)
