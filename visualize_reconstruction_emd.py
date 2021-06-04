from models import PotentialMapperRaysNN
from constants import *
from utils import MappingDataset, generate_dataset_from_strength
import matplotlib.pyplot as plt
import random
import torch

batch_size = 1
strengths = [0.01]
net = 'emd'
net_size = 1
train = True
model_name = 'EMD_2021-06-02 09_19_12.155170.pt'
skip_connections = False
model_path = MODELS_ROOT + model_name

power = 4

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

potential_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_potential.pt')
potential_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_potential.pt')

rays_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_rays.pt')
rays_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_rays.pt')

strength_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_strength.pt')
strength_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_strength.pt')

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

plt.show()
