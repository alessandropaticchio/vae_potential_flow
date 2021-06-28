from constants import *
from models import ConvVAE, ConvVAETest
from utils import MappingDataset, generate_dataset_from_strength
import torch

conditional = False
net_size = 1
batch_size = 500
train = False

potential_model_name = 'potential_VAE_[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]_2021-06-24 15_34_11.213966.pt'
rays_model_name = 'rays_VAE_[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]_2021-06-27 08_22_54.069847.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
strengths = STRENGTHS

potential_ae = ConvVAE(image_dim=POTENTIAL_IMAGE_SIZE, hidden_size=POTENTIAL_HIDDEN_SIZE,
                       latent_size=POTENTIAL_LATENT_SIZE,
                       image_channels=POTENTIAL_IMAGE_CHANNELS, net_size=net_size, conditional=conditional)
potential_ae.load_state_dict(torch.load(potential_model_path, map_location=torch.device('cpu')))
potential_ae.eval()

rays_ae = ConvVAE(image_dim=RAYS_IMAGE_SIZE, hidden_size=RAYS_HIDDEN_SIZE, latent_size=RAYS_LATENT_SIZE,
                  image_channels=RAYS_IMAGE_CHANNELS, net_size=net_size, conditional=conditional)
rays_ae.load_state_dict(torch.load(rays_model_path, map_location=torch.device('cpu')))
rays_ae.eval()

batch_idx = 0
flag = 0

if train:

    dataset_size = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_potential.pt').shape[0]

    while True:

        encoded_train_set_X = torch.empty(1, POTENTIAL_LATENT_SIZE * 2)
        encoded_train_set_y = torch.empty(1, RAYS_LATENT_SIZE * 2)

        start = batch_size * batch_idx
        end = batch_size * (batch_idx + 1)

        print('Batch start: {}, Batch end: {}'.format(start, end))

        potential_train_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_potential.pt')[
                                       start:end]
        rays_train_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_rays.pt')[start:end]
        strength_train_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_strength.pt')[
                                      start:end]

        if end > dataset_size:
            flag = 1

        potential_train_dataset, strength_train_dataset = generate_dataset_from_strength(potential_train_dataset_full,
                                                                                         strength_train_dataset_full,
                                                                                         strengths)
        rays_train_dataset, _ = generate_dataset_from_strength(rays_train_dataset_full, strength_train_dataset_full,
                                                               strengths)

        # Training set generation
        for i, sample in enumerate(potential_train_dataset):
            # Encoding
            potential_mean, potential_log_var = potential_ae.encode(sample.unsqueeze(0), strength_train_dataset[i])
            potential_sample_encoded = torch.cat((potential_mean, potential_log_var), 1)

            rays_mean, rays_log_var = rays_ae.encode(rays_train_dataset[i].unsqueeze(0), strength_train_dataset[i])
            rays_sample_encoded = torch.cat((rays_mean, rays_log_var), 1)

            encoded_train_set_X = torch.cat((encoded_train_set_X, potential_sample_encoded), 0)
            encoded_train_set_y = torch.cat((encoded_train_set_y, rays_sample_encoded), 0)

        batch_idx += 1

        #  First tensor is meaningless
        encoded_train_set_X = encoded_train_set_X[1:]
        encoded_train_set_y = encoded_train_set_y[1:]
        encoded_train_set = MappingDataset(x=encoded_train_set_X, y=encoded_train_set_y, d=strength_train_dataset)

        torch.save(encoded_train_set, DATA_ROOT + 'RP_images/mapped/training_{}.pt'.format(batch_idx))

        if flag:
            #  First tensor is meaningless
            encoded_train_set_X = encoded_train_set_X[1:]
            encoded_train_set_y = encoded_train_set_y[1:]
            encoded_train_set = MappingDataset(x=encoded_train_set_X, y=encoded_train_set_y, d=strength_train_dataset)

            torch.save(encoded_train_set, DATA_ROOT + 'RP_images/mapped/training_{}.pt'.format(batch_idx))

            # Freeing the memory
            encoded_train_set_X = None
            encoded_train_set_y = None
            encoded_train_set = None
            potential_train_dataset_full = None
            rays_train_dataset_full = None
            strength_train_dataset_full = None

            # Unifying data
            encoded_train_set_X = torch.empty(1, POTENTIAL_LATENT_SIZE * 2)
            encoded_train_set_y = torch.empty(1, RAYS_LATENT_SIZE * 2)
            encoded_strengths = torch.empty(1, 1)

            for i in range(1, batch_idx):
                chunk = torch.load(DATA_ROOT + 'RP_images/mapped/training_{}.pt'.format(i))
                encoded_train_set_X = torch.cat((encoded_train_set_X, chunk.X), 0)
                encoded_train_set_y = torch.cat((encoded_train_set_y, chunk.y), 0)
                encoded_strengths = torch.cat((encoded_strengths, chunk.D), 0)

            #  First tensor is meaningless
            encoded_train_set_X = encoded_train_set_X[1:]
            encoded_train_set_y = encoded_train_set_y[1:]
            encoded_strengths = encoded_strengths[1:]

            encoded_train_set = MappingDataset(x=encoded_train_set_X, y=encoded_train_set_y, d=encoded_strengths)
            torch.save(encoded_train_set, DATA_ROOT + 'RP_images/mapped/training.pt'.format(batch_idx))

            exit()

else:
    dataset_size = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_potential.pt').shape[0]
    encoded_test_set_X = torch.empty(1, POTENTIAL_LATENT_SIZE * 2)
    encoded_test_set_y = torch.empty(1, RAYS_LATENT_SIZE * 2)
    encoded_strengths = torch.empty(1, 1)

    while True:

        start = batch_size * batch_idx
        end = batch_size * (batch_idx + 1)

        print('Batch start: {}, Batch end: {}'.format(start, end))

        potential_test_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_potential.pt')[start:end]
        rays_test_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_rays.pt')[start:end]
        strength_test_dataset_full = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_strength.pt')[start:end]

        if end > dataset_size:
            flag = 1

        potential_test_dataset, strength_test_dataset = generate_dataset_from_strength(potential_test_dataset_full,
                                                                                       strength_test_dataset_full,
                                                                                       strengths)
        rays_test_dataset, _ = generate_dataset_from_strength(rays_test_dataset_full, strength_test_dataset_full,
                                                              strengths)

        # Test set generation
        for i, sample in enumerate(potential_test_dataset):
            # Encoding
            potential_mean, potential_log_var = potential_ae.encode(sample.unsqueeze(0), strength_test_dataset[i])
            potential_sample_encoded = torch.cat((potential_mean, potential_log_var), 1)

            rays_mean, rays_log_var = rays_ae.encode(rays_test_dataset[i].unsqueeze(0), strength_test_dataset[i])
            rays_sample_encoded = torch.cat((rays_mean, rays_log_var), 1)

            encoded_test_set_X = torch.cat((encoded_test_set_X, potential_sample_encoded), 0)
            encoded_test_set_y = torch.cat((encoded_test_set_y, rays_sample_encoded), 0)
            encoded_strengths = torch.cat((encoded_strengths, strength_test_dataset[i].unsqueeze(0)), 0)

        batch_idx += 1

        if flag:
            # First tensor is meaningless
            encoded_test_set_X = encoded_test_set_X[1:]
            encoded_test_set_y = encoded_test_set_y[1:]
            encoded_strengths = encoded_strengths[1:]
            encoded_test_set = MappingDataset(x=encoded_test_set_X, y=encoded_test_set_y, d=encoded_strengths)

            torch.save(encoded_test_set, DATA_ROOT + 'RP_images/mapped/test.pt')

            exit()
