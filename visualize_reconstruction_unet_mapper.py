from models import UNet_VAE
from constants import *
from utils import MyDataset
import matplotlib.pyplot as plt
import random
import torch

batch_size = 1

model_name = 'total_VAE__2021-03-02 18:23:05.490895.pt'
model_path = MODELS_ROOT + model_name

hidden_size = HIDDEN_SIZE
latent_size = LATENT_SIZE
vae = UNet_VAE(hidden_size=HIDDEN_SIZE, latent_size=latent_size)
vae.load_state_dict(torch.load(model_path))
vae.eval()


potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

# Compose datasets where X are potential samples, y are rays samples. The matching before samples from the same
# class is respected due to the generation of the subsets.
train_dataset = MyDataset(x=potential_train_dataset.data, y=rays_train_dataset.data)
test_dataset = MyDataset(x=potential_test_dataset.data, y=rays_test_dataset.data)

rand_sample_idx = random.randint(0, 160)
rand_sample = train_dataset.X[rand_sample_idx].unsqueeze(0)

rand_sample_prime, _, _ = vae(rand_sample)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 3, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Target')
plt.imshow(train_dataset.y[rand_sample_idx].detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
