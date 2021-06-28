from models import PotentialMapperRaysNN
from constants import *
from utils import MappingDataset, generate_dataset_from_strength
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms
import matplotlib.patches as mpatches

batch_size = 1
strengths = STRENGTHS
net = 'emd'
net_size = 1
train = True
model_name = 'EMD_2021-06-27 10_40_23.464265.pt'
model_path = MODELS_ROOT + model_name
skip_connections = False

power = 1

potential_image_size = POTENTIAL_IMAGE_SIZE
potential_image_channels = POTENTIAL_IMAGE_CHANNELS
potential_hidden_size = POTENTIAL_HIDDEN_SIZE
# potential_hidden_size = 4 * 47 * 47 * net_size
potential_latent_size = POTENTIAL_LATENT_SIZE
potential_image_channels = POTENTIAL_IMAGE_CHANNELS

rays_image_size = RAYS_IMAGE_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS
rays_hidden_size = RAYS_HIDDEN_SIZE
# rays_hidden_size = 4 * 47 * 47
rays_latent_size = RAYS_LATENT_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS

h_sizes = H_SIZES

vae = PotentialMapperRaysNN(potential_image_channels=potential_image_channels,
                            rays_image_channels=rays_image_channels,
                            potential_hidden_size=potential_hidden_size,
                            rays_hidden_size=rays_hidden_size,
                            potential_latent_size=potential_latent_size,
                            rays_latent_size=rays_latent_size,
                            h_sizes=h_sizes,
                            net_size=net_size,
                            skip_connections=skip_connections)

vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

potential_train_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_potential.pt')
potential_test_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_potential.pt')

rays_train_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_rays.pt')
rays_test_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_rays.pt')

strength_train_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_strength.pt')
strength_test_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_strength.pt')

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

if train:
    dataset = train_dataset
else:
    dataset = test_dataset

rand_sample_idx = random.randint(0, len(dataset))
rand_sample = dataset.X[rand_sample_idx].unsqueeze(0)

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
plt.imshow(dataset.y[rand_sample_idx].detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.figure()

rand_sample_prime = transforms.Compose([
    transforms.Grayscale(num_output_channels=1)
])(rand_sample_prime.squeeze(0))

target = transforms.Compose([
    transforms.Grayscale(num_output_channels=1)
])(dataset.y[rand_sample_idx].squeeze(0))

I_mean = torch.mean(rand_sample_prime, dim=1)
I_var = torch.var(rand_sample_prime, dim=1)

I_mean_real = torch.mean(target, dim=1)
I_var_real = torch.var(target, dim=1)

s = (I_var / I_mean) - 1
s_real = (I_var_real / I_mean_real) - 1

plt.title('Scintillation Index')
plt.plot(s.squeeze().detach().numpy(), label="reconstructed")
plt.plot(s_real.squeeze().detach().numpy(), label='original')

blue_patch = mpatches.Patch(color='blue', label='Reconstructed')
orange_patch = mpatches.Patch(color='orange', label='Original')
plt.legend(handles=[blue_patch, orange_patch])

plt.show()


