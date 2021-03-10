from models import ConvVAE
from constants import *
import matplotlib.pyplot as plt
import random
import torch
import itertools

batch_size = 1

dataset = 'potential'
model_name = 'potential_VAE__2021-03-10 20_41_23.106999.pt'
model_path = MODELS_ROOT + model_name

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    hidden_size = RAYS_HIDDEN_SIZE
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

# ae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)
ae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
             net_size=1)
ae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
ae.eval()

train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_' + dataset + '.pt')
test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_' + dataset + '.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

rand_sample_idx = random.randint(0, 40)
rand_sample = next(itertools.islice(test_loader, rand_sample_idx, None))

rand_sample_prime = ae(rand_sample[0].reshape(1, image_channels, image_size, image_size))[0]

plt.figure()

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 2, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
