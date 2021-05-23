import torch.optim as optim
from constants import *
from utils import MyDataset, MappingDataset, generate_dataset_from_strength
from training import train_unet_vae
from models import PotentialMapperRaysNN, ConvVAE, Mapper, ConvVAETest

net_size = 1
batch_size = 32
strengths = STRENGTHS
power = 4
gmm = len(STRENGTHS)

potential_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_potential.pt')
potential_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_potential.pt')

rays_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_rays.pt')
rays_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_rays.pt')

strength_train_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_strength.pt')
strength_test_dataset_full = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'test_strength.pt')

potential_train_dataset, strength_train_dataset = generate_dataset_from_strength(potential_train_dataset_full,
                                                                                 strength_train_dataset_full,
                                                                                 strengths)
potential_test_dataset, strength_test_dataset = generate_dataset_from_strength(potential_test_dataset_full,
                                                                               strength_test_dataset_full,
                                                                               strengths)

rays_train_dataset, _ = generate_dataset_from_strength(rays_train_dataset_full, strength_train_dataset_full,
                                                       strengths)
rays_test_dataset, _ = generate_dataset_from_strength(rays_test_dataset_full, strength_test_dataset_full,
                                                      strengths)

train_dataset = MappingDataset(x=potential_train_dataset, y=rays_train_dataset, d=strength_train_dataset)
test_dataset = MappingDataset(x=potential_test_dataset, y=rays_test_dataset, d=strength_test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

potential_image_size = POTENTIAL_IMAGE_SIZE
potential_image_channels = POTENTIAL_IMAGE_CHANNELS
potential_hidden_size = POTENTIAL_HIDDEN_SIZE
# potential_hidden_size = 4 * 47 * 47
potential_latent_size = POTENTIAL_LATENT_SIZE
potential_image_channels = POTENTIAL_IMAGE_CHANNELS

rays_image_size = RAYS_IMAGE_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS
rays_hidden_size = RAYS_HIDDEN_SIZE
# rays_hidden_size = 4 * 47 * 47
rays_latent_size = RAYS_LATENT_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS

h0 = POTENTIAL_LATENT_SIZE * 2
h1 = RAYS_LATENT_SIZE * 2

emd = PotentialMapperRaysNN(potential_image_channels=potential_image_channels,
                            rays_image_channels=rays_image_channels,
                            potential_hidden_size=potential_hidden_size,
                            rays_hidden_size=rays_hidden_size,
                            potential_latent_size=potential_latent_size,
                            rays_latent_size=rays_latent_size,
                            h_sizes=[h0, h1],
                            net_size=net_size)

#  Load VAEs
potential_model_name = 'potential_VAE_[0.01, 0.1, 0.2, 0.05, 0.07, 0.09, 0.03, 0.3]_2021-05-20 16_34_20.045217.pt'
rays_model_name = 'rays_VAE_[0.01, 0.03, 0.05, 0.1, 0.2, 0.07, 0.3]_2021-05-20 17_27_24.438039.pt'
mapper_model_name = 'Mapper_2021-05-23 08_14_48.519351.pt'
potential_model_path = MODELS_ROOT + potential_model_name
rays_model_path = MODELS_ROOT + rays_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name

potential_vae = ConvVAE(image_dim=potential_image_size, hidden_size=potential_hidden_size,
                            latent_size=potential_latent_size, image_channels=potential_image_channels,
                            net_size=net_size)
potential_vae.load_state_dict(torch.load(potential_model_path, map_location=torch.device('cpu')))

rays_vae = ConvVAE(image_dim=rays_image_size, hidden_size=rays_hidden_size, latent_size=rays_latent_size,
                       image_channels=rays_image_channels,
                       net_size=net_size)
rays_vae.load_state_dict(torch.load(rays_model_path, map_location=torch.device('cpu')))

mapper = Mapper(h_sizes=[h0, h1])
mapper.load_state_dict(torch.load(mapper_model_path, map_location=torch.device('cpu')))

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

# Decoder
emd.deconv1.weight = rays_vae.deconv1.weight
emd.deconv1.bias = rays_vae.deconv1.bias

emd.deconv2.weight = rays_vae.deconv2.weight
emd.deconv2.bias = rays_vae.deconv2.bias

lr = 1e-4
optimizer = optim.Adam(emd.parameters(), lr=lr)

recon_weight = 1.
kl_weight = 4.

train_unet_vae(net=emd, train_loader=train_loader, test_loader=test_loader, epochs=1, optimizer=optimizer, gmm=gmm,
               recon_weight=recon_weight, kl_weight=kl_weight, dataset='EMD', nn_type='conv', power=power)
