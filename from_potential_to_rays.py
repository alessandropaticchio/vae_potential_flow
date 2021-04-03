from models import ConvVAE, Mapper
from utils import StrengthDataset, generate_dataset_from_strength
from constants import *
import random
import matplotlib.pyplot as plt

potential_model_name = 'potential_VAE__2021-03-28 15_26_17.978434.pt'
rays_model_name = 'rays_VAE__2021-03-28 15_01_00.790258.pt'
mapper_model_name = 'Mapper_2021-03-28 15_44_57.245977.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name
train = True
strengths = [0.2, 0.3]

power = 4

potential_train_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'test_potential.pt')

rays_train_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'test_rays.pt')

strength_train_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'num=999/loaded_data/' + 'test_strength.pt')

potential_train_dataset, strength_train_dataset = generate_dataset_from_strength(potential_train_dataset,
                                                                                 strength_train_dataset,
                                                                                 strengths)
potential_test_dataset, strength_test_dataset = generate_dataset_from_strength(potential_test_dataset,
                                                                               strength_test_dataset,
                                                                               strengths)

rays_train_dataset, _ = generate_dataset_from_strength(rays_train_dataset, strength_train_dataset,
                                                       strengths)
rays_test_dataset, _ = generate_dataset_from_strength(rays_test_dataset, strength_test_dataset,
                                                      strengths)

potential_vae = ConvVAE(image_dim=POTENTIAL_IMAGE_SIZE, hidden_size=POTENTIAL_HIDDEN_SIZE,
                        latent_size=POTENTIAL_LATENT_SIZE,
                        image_channels=POTENTIAL_IMAGE_CHANNELS,
                        net_size=1)
potential_vae.load_state_dict(torch.load(potential_model_path, map_location=torch.device('cpu')))
potential_vae.eval()

rays_vae = ConvVAE(image_dim=RAYS_IMAGE_SIZE, hidden_size=RAYS_HIDDEN_SIZE, latent_size=RAYS_LATENT_SIZE,
                   image_channels=RAYS_IMAGE_CHANNELS,
                   net_size=1)
rays_vae.load_state_dict(torch.load(rays_model_path, map_location=torch.device('cpu')))
rays_vae.eval()

h0 = POTENTIAL_LATENT_SIZE * 2
h1 = RAYS_LATENT_SIZE * 2
h2 = RAYS_LATENT_SIZE * 2
h3 = RAYS_LATENT_SIZE * 2

mapper = Mapper(h_sizes=[h0, h1, h2, h3])
mapper.load_state_dict(torch.load(mapper_model_path, map_location=torch.device('cpu')))
mapper.eval()

if train:
    potential_dataset = potential_train_dataset
    rays_dataset = rays_train_dataset
else:
    potential_dataset = potential_test_dataset
    rays_train_dataset = rays_test_dataset

for i in range(1, 2):
    # Encoding
    rand_sample_idx = random.randint(1, 159)

    pic_potential_sample = potential_dataset.data[rand_sample_idx][0].unsqueeze(0)
    strength_potential_sample = potential_dataset.data[rand_sample_idx][1].unsqueeze(0)
    potential_mean, potential_log_var = potential_vae.encode(pic_potential_sample, strength_potential_sample)
    potential_sample_encoded = torch.cat((potential_mean, potential_log_var), 1)

    # Mapping
    mapping = mapper(potential_sample_encoded)

    rays_mean = mapping[:, :RAYS_LATENT_SIZE]
    rays_log_var = mapping[:, RAYS_LATENT_SIZE:]

    rays_sample_mapped = rays_vae.decode(rays_mean, rays_log_var)
    rays_sample_mapped = torch.pow(rays_sample_mapped, power)

    # Reconstructing the original image for comparison
    rays_sample_reconstructed = rays_vae(rays_dataset.data[rand_sample_idx].unsqueeze(0))[0]
    rays_sample_reconstructed = torch.pow(rays_sample_reconstructed, power)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Potential')
    plt.imshow(pic_potential_sample.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 2)
    plt.title('Original Rays')
    plt.imshow(rays_dataset[rand_sample_idx].permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 3)
    plt.title('Mapped Rays')
    plt.imshow(rays_sample_mapped.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 4)
    plt.title('VAE-Reconstructed Rays')
    plt.imshow(rays_sample_reconstructed.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.figure()
    pixel_val = int(POTENTIAL_IMAGE_SIZE / 5)
    plt.title('Projection along x = {}'.format(pixel_val))
    plt.plot(range(0, POTENTIAL_IMAGE_SIZE), rays_dataset[rand_sample_idx].squeeze()[0, :, pixel_val],
             label='Ground truth')
    plt.plot(range(0, POTENTIAL_IMAGE_SIZE), rays_sample_mapped.squeeze().detach().numpy()[0, :, pixel_val],
             label='Predicted')
    plt.legend(loc='best')

plt.show()
