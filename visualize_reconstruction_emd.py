from models import UNet_VAE, PotentialMapperRaysNN
from constants import *
from utils import MappingDataset, generate_dataset_from_strength
import matplotlib.pyplot as plt
import random
import torch

batch_size = 1
strengths = STRENGTHS
net = 'emd'
net_size = 2

model_name = 'EMD_VAE__2021-05-02 06_33_02.062099.pt'
model_path = MODELS_ROOT + model_name

power = 4

if net == 'unet':
    hidden_size = 128 * 147 * 147
    latent_size = POTENTIAL_LATENT_SIZE
    vae = UNet_VAE(hidden_size=hidden_size, latent_size=latent_size)
if net == 'emd':
    potential_image_size = POTENTIAL_IMAGE_SIZE
    potential_image_channels = POTENTIAL_IMAGE_CHANNELS
    # potential_hidden_size = POTENTIAL_HIDDEN_SIZE
    potential_hidden_size = 4 * 47 * 47 * net_size
    potential_latent_size = POTENTIAL_LATENT_SIZE
    potential_image_channels = POTENTIAL_IMAGE_CHANNELS

    rays_image_size = RAYS_IMAGE_SIZE
    rays_image_channels = RAYS_IMAGE_CHANNELS
    # rays_hidden_size = RAYS_HIDDEN_SIZE
    rays_hidden_size = 4 * 47 * 47
    rays_latent_size = RAYS_LATENT_SIZE
    rays_image_channels = RAYS_IMAGE_CHANNELS

    h0 = POTENTIAL_LATENT_SIZE * 2
    h1 = RAYS_LATENT_SIZE * 2
    h2 = RAYS_LATENT_SIZE * 2
    h3 = RAYS_LATENT_SIZE * 2

    vae = PotentialMapperRaysNN(potential_image_channels=potential_image_channels,
                                rays_image_channels=rays_image_channels,
                                potential_hidden_size=potential_hidden_size,
                                rays_hidden_size=rays_hidden_size,
                                potential_latent_size=potential_latent_size,
                                rays_latent_size=rays_latent_size,
                                h_sizes=[h0, h1, h2, h3],
                                net_size=net_size)

vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

potential_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_potential.pt')
potential_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_potential.pt')

rays_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_rays.pt')
rays_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_rays.pt')

strength_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_strength.pt')
strength_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_strength.pt')

potential_train_dataset, strength_train_dataset = generate_dataset_from_strength(potential_train_dataset_full,
                                                                                 strength_train_dataset_full,
                                                                                 strengths)
potential_test_dataset, strength_test_dataset = generate_dataset_from_strength(potential_test_dataset_full,
                                                                               strength_test_dataset_full,
                                                                               strengths)

rays_train_dataset, _ = generate_dataset_from_strength(rays_train_dataset_full, strength_train_dataset_full,
                                                       strengths)
rays_test_dataset, _ = generate_dataset_from_strength(rays_test_dataset_full, strength_test_dataset_full,
                                                      strengths)

train_dataset = MappingDataset(x=potential_train_dataset, y=rays_train_dataset, d=strength_train_dataset)
test_dataset = MappingDataset(x=potential_test_dataset, y=rays_test_dataset, d=strength_test_dataset)

rand_sample_idx = random.randint(0, 799)
rand_sample = train_dataset.X[rand_sample_idx].unsqueeze(0)

rand_sample_prime = vae(rand_sample)[0]
rand_sample_prime = torch.pow(rand_sample_prime, power)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 3, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime[0].squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Target')
plt.imshow(train_dataset.y[rand_sample_idx].detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
