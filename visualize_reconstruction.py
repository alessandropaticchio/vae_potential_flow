from models import ConvVAE, ConvPlainAE
from constants import *
import matplotlib.pyplot as plt
import random
import torch
import itertools

batch_size = 1

dataset = 'potential'
train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/test_' + dataset + '.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_name = 'Mapper_2020-12-23 18:22:18.775705.pt'
model_path = MODELS_ROOT + model_name

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    hidden_size = RAYS_HIDDEN_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    latent_size = int(hidden_size / 2)
    #vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)
    vae = ConvPlainAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)
else:
    image_size = POTENTIAL_IMAGE_SIZE
    hidden_size = POTENTIAL_HIDDEN_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    latent_size = int(hidden_size / 2)
    #vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)
    vae = ConvPlainAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)

vae.load_state_dict(torch.load(model_path))
vae.eval()

rand_sample_idx = random.randint(0, 500)
rand_sample = next(itertools.islice(train_loader, rand_sample_idx, None))

rand_sample_prime = vae(rand_sample[0].reshape(1, image_channels, image_size, image_size))[0]

plt.figure()

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.squeeze().detach().numpy(), cmap='gray')

'''for i in range(100):
    print(sum(rand_sample_prime[0][0][i]))'''

plt.show()
