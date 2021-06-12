from models import ConvVAE, ConvVAETest
from constants import *
from utils import StrengthDataset, generate_dataset_from_strength
import matplotlib.pyplot as plt
import random
import torch
import itertools
from torchvision import transforms
import matplotlib.patches as mpatches

batch_size = 1
n_forwards = 10

dataset = 'potential'
model_name = 'potential_VAE_[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]_2021-06-06 21_49_43.732777.pt'
model_path = MODELS_ROOT + model_name
conditional = False
net_size = 1

if dataset == 'potential':
    power = 1
    transform = True
else:
    power = 4
    transform = False

train = False
strengths = STRENGTHS

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    hidden_size = RAYS_HIDDEN_SIZE
    # hidden_size = 8 * 48 * 48
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    hidden_size = POTENTIAL_HIDDEN_SIZE
    # hidden_size = 8 * 48 * 48
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

ae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
             net_size=net_size, conditional=conditional)
# from models import ConvPlainAE
# ae = ConvPlainAE(image_channels=image_channels, net_size=net_size)
# ae = ConvVAETest(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
#                  net_size=net_size, conditional=conditional)
ae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
ae.eval()

pics_train_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_' + dataset + '.pt')
pics_test_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_' + dataset + '.pt')

strength_train_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_strength.pt')

pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)
pics_test_dataset, strength_test_dataset = generate_dataset_from_strength(pics_test_dataset, strength_test_dataset,
                                                                          strengths)

train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)
test_dataset = StrengthDataset(x=pics_test_dataset, d=strength_test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if train:
    loader = train_loader
else:
    loader = test_loader

rand_sample_idx = max(0, random.randint(0, len(loader)) - 1)
rand_sample, rand_strength = next(itertools.islice(loader, rand_sample_idx, None))

predicted_samples = []
for i in range(n_forwards):
    forward = ae(rand_sample, rand_strength)[0]
    # forward = ae(rand_sample)
    predicted_samples.append(forward)

rand_sample_prime = torch.mean(torch.stack(predicted_samples), dim=0)
# rand_sample_prime = torch.pow(rand_sample_prime, power)

if transform:

    rand_sample = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])(rand_sample.squeeze(0))

    rand_sample_prime = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])(rand_sample_prime.squeeze(0))

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(rand_sample.squeeze(), cmap='gray', vmin=0, vmax=1)

    plt.subplot(1, 2, 2)
    plt.title('Reconstruction')
    plt.imshow(rand_sample_prime.squeeze().detach().numpy(), cmap='gray', vmin=0, vmax=1)

else:
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.title('Reconstruction')
    plt.imshow(rand_sample_prime.squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

if dataset == 'rays':
    plt.figure()

    rand_sample_prime = transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
    ])(rand_sample_prime.squeeze(0))

    rand_sample = transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
    ])(rand_sample.squeeze(0))

    I_mean = torch.mean(rand_sample_prime, dim=1)
    I_var = torch.var(rand_sample_prime, dim=1)

    I_mean_real = torch.mean(rand_sample, dim=1)
    I_var_real = torch.var(rand_sample, dim=1)

    s = (I_var / I_mean) - 1
    s_real = (I_var_real / I_mean_real) - 1

    plt.title('Scintillation Index')
    plt.plot(s.squeeze().detach().numpy(), label="reconstructed")
    plt.plot(s_real.squeeze().detach().numpy(), label='original')

    blue_patch = mpatches.Patch(color='blue', label='Reconstructed')
    orange_patch = mpatches.Patch(color='orange', label='Original')
    plt.legend(handles=[blue_patch, orange_patch])

# if dataset == 'rays':
#     plt.figure()
#     pixel_val = RAYS_IMAGE_SIZE // 5
#     plt.title('Projection along x = {}'.format(pixel_val))
#     plt.plot(range(0, image_size), rand_sample.squeeze()[0, :, pixel_val], label='Ground truth')
#     plt.plot(range(0, image_size), rand_sample_prime.squeeze().detach().numpy()[0, :, pixel_val], label='Predicted')
#     plt.legend(loc='best')

plt.show()
