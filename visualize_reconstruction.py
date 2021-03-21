from models import ConvVAE, ConvVAETest
from constants import *
import matplotlib.pyplot as plt
import random
import torch
import itertools

batch_size = 1

dataset = 'rays'
model_name = 'rays_VAE__2021-03-21 13_30_02.272637.pt'
model_path = MODELS_ROOT + model_name
square = True

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    hidden_size = RAYS_HIDDEN_SIZE
    # hidden_size = 4 * 146 * 146
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

ae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
             net_size=1)
ae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
ae.eval()

train_dataset = torch.load(DATA_ROOT + 'D=0.3 num=999/loaded_data/' + 'training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'D=0.3 num=999/loaded_data/' + 'test_' + dataset + '.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

rand_sample_idx = random.randint(0, 80)
rand_sample = next(itertools.islice(train_loader, rand_sample_idx, None))

rand_sample_prime = ae(rand_sample[0].reshape(1, image_channels, image_size, image_size))[0]

if square:
    rand_sample_prime = rand_sample_prime.square()

plt.figure()

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 2, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
