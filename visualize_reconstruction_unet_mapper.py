from models import UNet_VAE
from constants import *
from utils import MyDataset
import matplotlib.pyplot as plt
import random
import torch

batch_size = 1

model_name = 'total_VAE__2021-03-09 18:26:41.501571.pt'
model_path = MODELS_ROOT + model_name

hidden_size = 128 * 147 * 147
latent_size = LATENT_SIZE
vae = UNet_VAE(hidden_size=hidden_size, latent_size=latent_size)
vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

data_path = DATA_ROOT + '/real_data/'

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

train_dataset = MyDataset(x=potential_train_dataset, y=rays_train_dataset)
test_dataset = MyDataset(x=potential_test_dataset, y=rays_test_dataset)

train_dataset.y = train_dataset.y[0, :, :, :].unsqueeze(0)
train_dataset.X = train_dataset.X[0, :, :, :].unsqueeze(0)

rand_sample_idx = random.randint(0, 159)
rand_sample = train_dataset.X[0].unsqueeze(0)

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
plt.imshow(train_dataset.y[0].detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
