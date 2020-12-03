from fashion_mnist_downloader import train_dataset, test_dataset
from models import LinearVAE
from constants import *
import matplotlib.pyplot as plt
import random
import torch
import itertools

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_name = 'Fashion_MNIST_VAE_2020-12-02 12:02:42.435081.pt'
model_path = MODELS_ROOT + model_name

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
