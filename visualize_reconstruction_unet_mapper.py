from models import UNet_VAE
from constants import *
from utils import MyDataset
import matplotlib.pyplot as plt
import random
import torch

batch_size = 1

model_name = 'total_VAE__2021-03-06 16:12:56.877493.pt'
model_path = MODELS_ROOT + model_name

hidden_size = HIDDEN_SIZE
latent_size = LATENT_SIZE
vae = UNet_VAE(hidden_size=HIDDEN_SIZE, latent_size=latent_size)
vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

data_path = DATA_ROOT + '/real_data/'

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

whole_train_dataset = MyDataset(x=potential_train_dataset, y=rays_train_dataset)
whole_test_dataset = MyDataset(x=potential_test_dataset, y=rays_test_dataset)

whole_train_dataset.y = whole_train_dataset.y[1, :,:, :].unsqueeze(0)
whole_train_dataset.X = whole_train_dataset.X[1, :,:, :].unsqueeze(0)

rand_sample_idx = random.randint(0, 159)
rand_sample = whole_train_dataset.X[0].unsqueeze(0)

a = rand_sample.squeeze().permute(1, 2, 0)

rand_sample_prime = vae(rand_sample)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 3, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime[0].squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Target')
plt.imshow(whole_train_dataset.y[0].detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
