from constants import *
from models import ConvPlainAE
from utils import MyDataset
import torch

potential_model_name = 'AE_potential_2021-01-08 17:34:36.296097.pt'
rays_model_name = 'AE_rays_2021-01-08 17:13:55.022048.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name

mapper_type = 'conv'

#potential_ae = ConvPlainAE(image_dim=POTENTIAL_IMAGE_SIZE, image_channels=POTENTIAL_IMAGE_CHANNELS)
potential_ae.load_state_dict(torch.load(potential_model_path))
potential_ae.eval()

#rays_ae = ConvPlainAE(image_dim=RAYS_IMAGE_SIZE, image_channels=RAYS_IMAGE_CHANNELS)
rays_ae.load_state_dict(torch.load(rays_model_path))
rays_ae.eval()

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

if mapper_type == 'conv':
    encoded_train_set_X = torch.empty(1, POTENTIAL_IMAGE_CHANNELS,
                                      POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE)
    encoded_train_set_y = torch.empty(1, RAYS_IMAGE_CHANNELS,
                                      RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE)

    encoded_test_set_X = torch.empty(1, POTENTIAL_IMAGE_CHANNELS,
                                      POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE)
    encoded_test_set_y = torch.empty(1, RAYS_IMAGE_CHANNELS,
                                      RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE)
else:
    encoded_train_set_X = torch.empty(1, POTENTIAL_ENCODED_SIZE)
    encoded_train_set_y = torch.empty(1, RAYS_ENCODED_SIZE)

    encoded_test_set_X = torch.empty(1, POTENTIAL_ENCODED_SIZE)
    encoded_test_set_y = torch.empty(1, RAYS_ENCODED_SIZE)

# Training set generation
for i, sample in enumerate(potential_train_dataset):
    # Encoding
    potential_sample_encoded = potential_ae.encode(sample.unsqueeze(0))
    rays_sample_encoded = rays_ae.encode(rays_train_dataset[i].unsqueeze(0))

    # Flattening
    if mapper_type != 'conv':
        potential_sample_encoded = potential_sample_encoded.view(potential_sample_encoded.size(0), -1)
        rays_sample_encoded = rays_sample_encoded.view(rays_sample_encoded.size(0), -1)

    encoded_train_set_X = torch.cat((encoded_train_set_X, sample.unsqueeze(0)), 0)
    encoded_train_set_y = torch.cat((encoded_train_set_y, rays_train_dataset[i].unsqueeze(0)), 0)

# Test set generation
for i, sample in enumerate(potential_test_dataset):
    # Encoding
    potential_sample_encoded = potential_ae.encode(sample.unsqueeze(0))
    rays_sample_encoded = rays_ae.encode(rays_test_dataset[i].unsqueeze(0))

    # Flattening
    if mapper_type != 'conv':
        potential_sample_encoded = potential_sample_encoded.view(potential_sample_encoded.size(0), -1)
        rays_sample_encoded = rays_sample_encoded.view(rays_sample_encoded.size(0), -1)

    encoded_test_set_X = torch.cat((encoded_test_set_X, sample.unsqueeze(0)), 0)
    encoded_test_set_y = torch.cat((encoded_test_set_y, rays_train_dataset[i].unsqueeze(0)), 0)

#  First tensor is meaningless
encoded_train_set_X = encoded_train_set_X[1:]
encoded_train_set_y = encoded_train_set_y[1:]
encoded_test_set_X = encoded_test_set_X[1:]
encoded_test_set_y = encoded_test_set_y[1:]

encoded_test_set = MyDataset(x=encoded_test_set_X, y=encoded_test_set_y)
encoded_train_set = MyDataset(x=encoded_train_set_X, y=encoded_train_set_y)

torch.save(encoded_train_set, DATA_ROOT + '/mapped/training.pt')
torch.save(encoded_test_set, DATA_ROOT + '/mapped/test.pt')
