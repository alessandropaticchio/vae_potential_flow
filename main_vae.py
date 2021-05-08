from training import train_vae
from models import ConvVAE, ConvVAETest
from constants import *
from utils import StrengthDataset, generate_dataset_from_strength
import torch
import torch.optim as optim

batch_size = 64
vae_type = 'conv'
conditional = False
dataset = 'potential'
epochs = 150
strengths = STRENGTHS

pics_train_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_' + dataset + '.pt')
pics_test_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_' + dataset + '.pt')

strength_train_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_strength.pt')

pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)
pics_test_dataset, strength_test_dataset = generate_dataset_from_strength(pics_test_dataset, strength_test_dataset,
                                                                          strengths)

train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)
test_dataset = StrengthDataset(x=pics_test_dataset, d=strength_test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    # hidden_size = RAYS_HIDDEN_SIZE
    hidden_size = 4 * 46 * 46
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    # hidden_size = POTENTIAL_HIDDEN_SIZE
    hidden_size = 4 * 46 * 46
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

vae = ConvVAETest(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
                  net_size=2, conditional=conditional)

lr = 1e-3
optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=0)

if dataset == 'potential':
    power = 1
else:
    power = 4

recon_weight = 1.
kl_weight = 1.
reg_weight = 0.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset, nn_type=vae_type, is_L1=False, power=power,
          desc=strengths, reg_weight=reg_weight)
