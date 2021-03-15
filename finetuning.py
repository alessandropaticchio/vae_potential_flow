import torch.optim as optim
from constants import *
from utils import MyDataset
from training import train_unet_vae
from models import PotentialMapperRaysNN, ConvVAE, Mapper

data_path = DATA_ROOT + '/real_data/'

potential_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_potential.pt')
potential_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_potential.pt')
rays_train_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'training_rays.pt')
rays_test_dataset = torch.load(DATA_ROOT + 'DATA21.2.18/loaded_data/' + 'test_rays.pt')

train_dataset = MyDataset(x=potential_train_dataset, y=rays_train_dataset)
test_dataset = MyDataset(x=potential_test_dataset, y=rays_test_dataset)

batch_size = 8
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

potential_image_size = RAYS_IMAGE_SIZE
potential_image_channels = RAYS_IMAGE_CHANNELS
potential_hidden_size = RAYS_HIDDEN_SIZE
potential_latent_size = RAYS_LATENT_SIZE
potential_image_channels = RAYS_IMAGE_CHANNELS

rays_image_size = RAYS_IMAGE_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS
rays_hidden_size = RAYS_HIDDEN_SIZE
rays_latent_size = RAYS_LATENT_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS

h0 = POTENTIAL_LATENT_SIZE * 2
h1 = RAYS_LATENT_SIZE * 2
h2 = RAYS_LATENT_SIZE * 2
h3 = RAYS_LATENT_SIZE * 2

emd = PotentialMapperRaysNN(potential_image_channels=potential_image_channels,
                            rays_image_channels=rays_image_channels,
                            potential_hidden_size=potential_hidden_size,
                            rays_hidden_size=rays_hidden_size,
                            potential_latent_size=potential_latent_size,
                            rays_latent_size=rays_latent_size,
                            h_sizes=[h0, h1, h2, h3],
                            net_size=1)

#  Load VAEs
potential_model_name = 'potential_VAE__2021-03-09 18:07:17.864741.pt'
rays_model_name = 'rays_VAE__2021-03-09 18:11:30.087612.pt'
mapper_model_name = 'Mapper_2021-03-09 18:16:31.732229.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name

potential_vae = ConvVAE(image_dim=potential_image_size, hidden_size=potential_hidden_size,
                        latent_size=potential_latent_size, image_channels=potential_image_channels,
                        net_size=1)
potential_vae.load_state_dict(torch.load(potential_model_path))

rays_vae = ConvVAE(image_dim=rays_image_size, hidden_size=rays_hidden_size, latent_size=rays_latent_size,
                   image_channels=rays_image_channels,
                   net_size=1)
rays_vae.load_state_dict(torch.load(rays_model_path))

mapper = Mapper(h_sizes=[h0, h1, h2, h3])
mapper.load_state_dict(torch.load(mapper_model_path))

# Initializing emd's encoder as potential encoder and decoder as rays decoder
# Encoder
emd.conv1.weight = potential_vae.conv1.weight
emd.conv1.bias = potential_vae.conv1.bias

emd.conv2.weight = potential_vae.conv2.weight
emd.conv2.bias = potential_vae.conv2.bias

emd.encoder_mean.weight = potential_vae.encoder_mean.weight
emd.encoder_mean.bias = potential_vae.encoder_mean.bias

emd.encoder_logvar.weight = potential_vae.encoder_logvar.weight
emd.encoder_logvar.bias = potential_vae.encoder_logvar.bias

emd.fc.weight = potential_vae.fc.weight
emd.fc.bias = potential_vae.fc.bias

# Mapper
emd.mapper_layers[0].weight = mapper.layers[0].weight
emd.mapper_layers[0].bias = mapper.layers[0].bias

emd.mapper_layers[1].weight = mapper.layers[1].weight
emd.mapper_layers[1].bias = mapper.layers[1].bias

emd.mapper_layers[2].weight = mapper.layers[2].weight
emd.mapper_layers[2].bias = mapper.layers[2].bias

# Decoder
emd.deconv1.weight = potential_vae.deconv1.weight
emd.deconv1.bias = potential_vae.deconv1.bias

emd.deconv2.weight = potential_vae.deconv2.weight
emd.deconv2.bias = potential_vae.deconv2.bias

lr = 1e-4
optimizer = optim.Adam(emd.parameters(), lr=lr)

recon_weight = 1.
kl_weight = 1.

train_unet_vae(net=emd, train_loader=train_loader, test_loader=test_loader, epochs=300, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset='EMD', nn_type='conv')