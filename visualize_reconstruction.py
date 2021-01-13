from models import LinearVAE, ConvVAE
from constants import *
import matplotlib.pyplot as plt
import random
import torch
import itertools

vae_type = 'conv'
dataset = 'MNIST'

if dataset == 'MNIST':
    from mnist_downloader import train_dataset, test_dataset
else:
    from fashion_mnist_downloader import train_dataset, test_dataset

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_name = 'MNIST_VAE_2021-01-11 22:54:43.183412.pt'
model_path = MODELS_ROOT + model_name

if vae_type == 'conv':
    hidden_size = 1568
    vae = ConvVAE(image_dim=28, hidden_size=hidden_size, latent_size=int(hidden_size / 2), image_channels=1)
else:
    vae = LinearVAE()

vae.load_state_dict(torch.load(model_path))
vae.eval()

rand_sample_idx = random.randint(0, 100)
rand_sample = next(itertools.islice(test_loader, rand_sample_idx, None))

rand_sample_prime = vae(rand_sample[0])[0]

plt.figure()
plt.imshow(rand_sample[0].reshape(28, 28).detach().numpy())

plt.figure()
plt.imshow(rand_sample_prime.reshape(28, 28).detach().numpy())

plt.show()
