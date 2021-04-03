from models import ConvVAE, ConvVAETest
from constants import *
from utils import StrengthDataset, generate_dataset_from_strength
import matplotlib.pyplot as plt
import random
import torch
import itertools

batch_size = 1

dataset = 'rays'
model_name = 'rays_VAE_[0.2, 0.3]_1617431982.820919.pt'
model_path = MODELS_ROOT + model_name
power = 1
train = True
strengths = [0.2, 0.3]

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

pics_train_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'training_' + dataset + '.pt')
pics_test_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'test_' + dataset + '.pt')

strength_train_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'test_strength.pt')

pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)
pics_test_dataset, strength_test_dataset = generate_dataset_from_strength(pics_test_dataset, strength_test_dataset,
                                                                          strengths)

train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)
test_dataset = StrengthDataset(x=pics_test_dataset, d=strength_test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if train:
    loader = train_loader
else:
    loader = test_loader

rand_sample_idx = random.randint(0, len(loader))
rand_sample, rand_strength = next(itertools.islice(loader, rand_sample_idx, None))

rand_sample_prime = ae(rand_sample[0].reshape(1, image_channels, image_size, image_size), rand_strength)[0]
rand_sample_prime = torch.pow(rand_sample_prime, power)

plt.figure()

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 2, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

if dataset == 'rays':
    plt.figure()
    pixel_val = RAYS_IMAGE_SIZE // 5
    plt.title('Projection along x = {}'.format(pixel_val))
    plt.plot(range(0, image_size), rand_sample.squeeze()[0, :, pixel_val], label='Ground truth')
    plt.plot(range(0, image_size), rand_sample_prime.squeeze().detach().numpy()[0, :, pixel_val], label='Predicted')
    plt.legend(loc='best')

plt.show()
