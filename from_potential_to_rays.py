from models import ConvVAE, Mapper
from constants import *
import random
import matplotlib.pyplot as plt

potential_model_name = 'potential_VAE__2021-03-09 18:07:17.864741.pt'
rays_model_name = 'rays_VAE__2021-03-09 18:11:30.087612.pt'
mapper_model_name = 'Mapper_2021-03-09 18:16:31.732229.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')

rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

#  Fetch first sample
potential_train_dataset = potential_train_dataset[0, :, :, :].unsqueeze(0)
rays_train_dataset = rays_train_dataset[0, :, :, :].unsqueeze(0)

potential_vae = ConvVAE(image_dim=POTENTIAL_IMAGE_SIZE, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE,
                        image_channels=POTENTIAL_IMAGE_CHANNELS,
                        net_size=1)
potential_vae.load_state_dict(torch.load(potential_model_path))
potential_vae.eval()

rays_vae = ConvVAE(image_dim=RAYS_IMAGE_SIZE, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE,
                   image_channels=RAYS_IMAGE_CHANNELS,
                   net_size=1)
rays_vae.load_state_dict(torch.load(rays_model_path))
rays_vae.eval()

# mapper = Mapper(h_sizes=[POTENTIAL_ENCODED_SIZE, POTENTIAL_ENCODED_SIZE, RAYS_ENCODED_SIZE, RAYS_ENCODED_SIZE])
mapper = Mapper(h_sizes=[LATENT_SIZE * 2, LATENT_SIZE * 2, LATENT_SIZE * 2, LATENT_SIZE * 2])
mapper.load_state_dict(torch.load(mapper_model_path))
mapper.eval()

for i in range(1, 10):
    # Encoding
    rand_sample_idx = random.randint(1, 799)

    potential_sample = potential_train_dataset.data[rand_sample_idx].unsqueeze(0)
    potential_mean, potential_log_var = potential_vae.encode(potential_sample)
    potential_sample_encoded = torch.cat((potential_mean, potential_log_var), 1)

    # Mapping
    mapping = mapper(potential_sample_encoded)

    rays_mean = mapping[:, :LATENT_SIZE]
    rays_log_var = mapping[:, LATENT_SIZE:]

    rays_sample_mapped = rays_vae.decode(rays_mean, rays_log_var)

    # Reconstructing the original image for comparison
    rays_sample_reconstructed = rays_vae(rays_train_dataset.data[rand_sample_idx].unsqueeze(0))[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Potential')
    plt.imshow(potential_sample.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 2)
    plt.title('Original Rays')
    plt.imshow(rays_train_dataset[rand_sample_idx].permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 3)
    plt.title('Mapped Rays')
    plt.imshow(rays_sample_mapped.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 4)
    plt.title('VAE-Reconstructed Rays')
    plt.imshow(rays_sample_reconstructed.squeeze(0).permute(1, 2, 0).detach().numpy())

plt.show()
