from models import ConvPlainAE, Mapper, ConvMapper
from constants import *
import random
import matplotlib.pyplot as plt

potential_model_name = 'AE_potential_2021-01-02 17:43:39.774670.pt'
rays_model_name = 'AE_rays_2021-01-04 11:50:24.176853.pt'
mapper_model_name = 'Mapper_2021-01-04 12:23:54.121672.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name

mapper_type = 'conv'

potential_train_dataset = torch.load(DATA_ROOT + 'real_data/' + POTENTIAL_ROOT + 'training_' + 'potential' + '.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'real_data/' + POTENTIAL_ROOT + 'test_' + 'potential' + '.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'real_data/' + RAYS_ROOT + 'training_' + 'rays' + '.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'real_data/' + RAYS_ROOT + 'test_' + 'rays' + '.pt')

potential_ae = ConvPlainAE(image_dim=POTENTIAL_IMAGE_SIZE, image_channels=POTENTIAL_IMAGE_CHANNELS)
potential_ae.load_state_dict(torch.load(potential_model_path))
potential_ae.eval()

rays_ae = ConvPlainAE(image_dim=RAYS_IMAGE_SIZE, image_channels=RAYS_IMAGE_CHANNELS)
rays_ae.load_state_dict(torch.load(rays_model_path))
rays_ae.eval()

# mapper = Mapper(h_sizes=[POTENTIAL_ENCODED_SIZE, POTENTIAL_ENCODED_SIZE, RAYS_ENCODED_SIZE, RAYS_ENCODED_SIZE])
mapper = ConvMapper(n_layers=4, potential_encoded_size=POTENTIAL_ENCODED_IMAGE_SIZE[1], rays_encoded_size=RAYS_ENCODED_IMAGE_SIZE[1])
mapper.load_state_dict(torch.load(mapper_model_path))
mapper.eval()

for i in range(1, 10):
    # Encoding
    rand_sample_idx = random.randint(1, 799)
    potential_sample = potential_train_dataset.data[rand_sample_idx].unsqueeze(0)
    potential_sample_encoded = potential_ae.encode(potential_sample)

    # Flattening
    if mapper_type != 'conv':
        potential_sample_encoded = potential_sample_encoded.view(potential_sample_encoded.size(0), -1)

    # Mapping
    mapping = mapper(potential_sample_encoded)

    # Unflattening
    if mapper_type != 'conv':
        mapping = mapping.view(1, 8, 25, 25)
    rays_sample_mapped = rays_ae.decode(mapping)

    # Decoding the original image for comparison
    rays_sample_decoded = rays_ae(rays_train_dataset.data[rand_sample_idx].unsqueeze(0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Potential')
    plt.imshow(potential_sample.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 2)
    plt.title('Original Rays')
    plt.imshow(rays_train_dataset[rand_sample_idx].permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 3)
    plt.title('Decoded Rays')
    plt.imshow(rays_sample_decoded.squeeze(0).permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 4, 4)
    plt.title('Mapped Rays')
    plt.imshow(rays_sample_mapped.squeeze(0).permute(1, 2, 0).detach().numpy())

plt.show()
