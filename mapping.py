from constants import *
from models import ConvVAE
from utils import MappingDataset, generate_dataset_from_strength
import torch

conditional = False

potential_model_name = 'potential_VAE_[0.01, 0.3]_2021-04-23 13_24_56.913723.pt'
rays_model_name = 'rays_VAE_[0.01, 0.3]_2021-04-23 13_14_47.903668.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
strengths = [0.01, 0.3]

potential_ae = ConvVAE(image_dim=POTENTIAL_IMAGE_SIZE, hidden_size=POTENTIAL_HIDDEN_SIZE, latent_size=POTENTIAL_LATENT_SIZE,
                       image_channels=POTENTIAL_IMAGE_CHANNELS, conditional=conditional)
potential_ae.load_state_dict(torch.load(potential_model_path, map_location=torch.device('cpu')))
potential_ae.eval()

rays_ae = ConvVAE(image_dim=RAYS_IMAGE_SIZE, hidden_size=RAYS_HIDDEN_SIZE, latent_size=RAYS_LATENT_SIZE,
                  image_channels=RAYS_IMAGE_CHANNELS, conditional=conditional)
rays_ae.load_state_dict(torch.load(rays_model_path, map_location=torch.device('cpu')))
rays_ae.eval()

potential_train_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_potential.pt')

rays_train_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_rays.pt')

strength_train_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_strength.pt')

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

encoded_train_set_X = torch.empty(1, POTENTIAL_LATENT_SIZE * 2)
encoded_train_set_y = torch.empty(1, RAYS_LATENT_SIZE * 2)

encoded_test_set_X = torch.empty(1, POTENTIAL_LATENT_SIZE * 2)
encoded_test_set_y = torch.empty(1, RAYS_LATENT_SIZE * 2)

# Training set generation
for i, sample in enumerate(potential_train_dataset):
    # Encoding
    potential_mean, potential_log_var = potential_ae.encode(sample.unsqueeze(0),strength_train_dataset[i])
    potential_sample_encoded = torch.cat((potential_mean, potential_log_var), 1)

    rays_mean, rays_log_var = rays_ae.encode(rays_train_dataset[i].unsqueeze(0), strength_train_dataset[i])
    rays_sample_encoded = torch.cat((rays_mean, rays_log_var), 1)

    encoded_train_set_X = torch.cat((encoded_train_set_X, potential_sample_encoded), 0)
    encoded_train_set_y = torch.cat((encoded_train_set_y, rays_sample_encoded), 0)

# Test set generation
for i, sample in enumerate(potential_test_dataset):
    # Encoding
    potential_mean, potential_log_var = potential_ae.encode(sample.unsqueeze(0), strength_test_dataset[i])
    potential_sample_encoded = torch.cat((potential_mean, potential_log_var), 1)

    rays_mean, rays_log_var = rays_ae.encode(rays_test_dataset[i].unsqueeze(0), strength_test_dataset[i])
    rays_sample_encoded = torch.cat((rays_mean, rays_log_var), 1)

    encoded_test_set_X = torch.cat((encoded_test_set_X, potential_sample_encoded), 0)
    encoded_test_set_y = torch.cat((encoded_test_set_y, rays_sample_encoded), 0)

#  First tensor is meaningless
encoded_train_set_X = encoded_train_set_X[1:]
encoded_train_set_y = encoded_train_set_y[1:]
encoded_test_set_X = encoded_test_set_X[1:]
encoded_test_set_y = encoded_test_set_y[1:]

encoded_test_set = MappingDataset(x=encoded_test_set_X, y=encoded_test_set_y, d=strength_train_dataset)
encoded_train_set = MappingDataset(x=encoded_train_set_X, y=encoded_train_set_y, d=strength_test_dataset)

torch.save(encoded_train_set, DATA_ROOT + 'D=0.3 num=999_unzipped/mapped/training.pt')
torch.save(encoded_test_set, DATA_ROOT + 'D=0.3 num=999_unzipped/mapped/test.pt')
