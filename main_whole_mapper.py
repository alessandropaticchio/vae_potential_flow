import torch.optim as optim
from constants import *
from utils import MyDataset
from training import train_mapper
from models import ConvWholeMapper, ConvPlainAE

data_path = DATA_ROOT + '/real_data/'

potential_train_dataset = torch.load(data_path + POTENTIAL_ROOT + 'training_potential.pt')
potential_test_dataset = torch.load(data_path + POTENTIAL_ROOT + 'test_potential.pt')

rays_train_dataset = torch.load(data_path + RAYS_ROOT + 'training_rays.pt')
rays_test_dataset = torch.load(data_path + RAYS_ROOT + 'test_rays.pt')

whole_train_dataset = MyDataset(x=potential_train_dataset, y=rays_train_dataset)
whole_test_dataset = MyDataset(x=potential_test_dataset, y=rays_test_dataset)

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=whole_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=whole_test_dataset, batch_size=batch_size, shuffle=False)

whole_mapper = ConvWholeMapper(potential_encoded_size=POTENTIAL_ENCODED_IMAGE_SIZE[1],
                               rays_encoded_size=RAYS_ENCODED_IMAGE_SIZE[1])

#  Load AEs
potential_model_name = 'AE_potential_2021-01-05 15:44:23.618717.pt'
rays_model_name = 'AE_rays_2021-01-05 18:48:05.829239.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name

potential_ae = ConvPlainAE(image_channels=POTENTIAL_IMAGE_CHANNELS)
potential_ae.load_state_dict(torch.load(potential_model_path))
potential_ae.eval()

rays_ae = ConvPlainAE(image_channels=RAYS_IMAGE_CHANNELS)
rays_ae.load_state_dict(torch.load(rays_model_path))
rays_ae.eval()

# Initializing whole_mapper's encoder as potential encoder and decoder as rays decoder
whole_mapper.conv1.weight = potential_ae.conv1.weight
whole_mapper.conv2.weight = rays_ae.conv2.weight
whole_mapper.conv3.weight = rays_ae.conv3.weight

# Freezing encoder and decoder
whole_mapper.conv1.weight.requires_grad = False
whole_mapper.conv2.weight.requires_grad = False
whole_mapper.conv3.weight.requires_grad = False

lr = 1e-3
optimizer = optim.Adam(whole_mapper.parameters(), lr=lr)

train_mapper(net=whole_mapper, train_loader=train_loader, test_loader=test_loader, epochs=1, optimizer=optimizer)

# Unfreezing and setting a low learning rate
whole_mapper.conv1.weight.requires_grad = True
whole_mapper.conv2.weight.requires_grad = True
whole_mapper.conv3.weight.requires_grad = True

optimizer = optim.Adam(whole_mapper.parameters(), lr=lr / 30)

train_mapper(net=whole_mapper, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer)
