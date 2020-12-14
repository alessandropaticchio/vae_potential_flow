from models import ConvVAE
from constants import *
import matplotlib.pyplot as plt
import random
import torch
import itertools

batch_size = 1

dataset = 'potential'
train_dataset = torch.load(DATA_ROOT + 'fake_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'fake_data/' + dataset + '_pic_data/test_' + dataset + '.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_name = 'potential_VAE_2020-12-13 11:55:48.693892.pt'
model_path = MODELS_ROOT + model_name

image_size = POTENTIAL_IMAGE_SIZE
hidden_size = POTENTIAL_HIDDEN_SIZE
latent_size = int(hidden_size / 2)
vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size)
vae.load_state_dict(torch.load(model_path))
vae.eval()

rand_sample_idx = random.randint(0, 100)
rand_sample = next(itertools.islice(train_loader, rand_sample_idx, None))

rand_sample_prime = vae(rand_sample[0].reshape(1, 3, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))[0]

plt.figure()
plt.imshow(rand_sample[0].permute(1, 2, 0))

plt.figure()
plt.imshow(rand_sample_prime.squeeze().permute(1, 2, 0).detach())

plt.show()
