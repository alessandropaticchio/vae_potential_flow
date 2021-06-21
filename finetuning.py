import torch.optim as optim
from constants import *
from utils import MyDataset, MappingDataset, generate_dataset_from_strength
from training import train_unet_vae
from models import PotentialMapperRaysNN, ConvVAE, Mapper, ConvVAETest

net_size = 1
batch_size = 32
strengths = STRENGTHS
power = 4
epochs = 30
gmm = len(STRENGTHS)
kl_annealing = False
skip_connections = False

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
#  rays_hidden_size = 4 * 47 * 47
rays_latent_size = RAYS_LATENT_SIZE
rays_image_channels = RAYS_IMAGE_CHANNELS

h_sizes = H_SIZES

emd = PotentialMapperRaysNN(potential_image_channels=potential_image_channels,
                            rays_image_channels=rays_image_channels,
                            potential_hidden_size=potential_hidden_size,
                            rays_hidden_size=rays_hidden_size,
                            potential_latent_size=potential_latent_size,
                            rays_latent_size=rays_latent_size,
                            h_sizes=h_sizes,
                            net_size=net_size,
                            skip_connections=skip_connections)

#  Load VAEs
potential_model_name = 'potential_VAE_[0.01, 0.1, 0.2, 0.05, 0.07, 0.03, 0.3]_2021-05-20 15_58_12.695847.pt'
rays_model_name = 'rays_VAE_[0.01, 0.03, 0.05, 0.1, 0.2, 0.07, 0.3]_2021-05-20 17_27_24.438039.pt'
mapper_model_name = 'Mapper_2021-06-02 08_08_49.940747.pt'
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

mapper = Mapper(h_sizes=h_sizes)
mapper.load_state_dict(torch.load(mapper_model_path, map_location=torch.device('cpu')))

emd_dict = emd.state_dict()

potential_vae_dict = potential_vae.state_dict()
encoder_keys = ['conv1.weight', 'conv1.bias',
                'conv2.weight', 'conv2.bias',
                'encoder_mean.weight', 'encoder_mean.bias',
                'encoder_logvar.weight', 'encoder_logvar.bias']
encoder_model_dict = {k: v for k, v in potential_vae_dict.items() if k in encoder_keys}

mapper_dict = mapper.state_dict()
mapper_keys = ['layers.0.weight', 'layers.0.bias']
mapper_model_dict = {k: v for k, v in mapper_dict.items() if k in mapper_keys}

rays_vae_dict = rays_vae.state_dict()
decoder_keys = ['fc.weight', 'fc.bias',
                'deconv1.weight', 'deconv1.bias',
                'deconv2.weight', 'deconv2.bias']
decoder_model_dict = {k: v for k, v in rays_vae_dict.items() if k in decoder_keys}

emd_dict.update(encoder_model_dict)
emd_dict.update(mapper_model_dict)
emd_dict.update(decoder_model_dict)
emd.load_state_dict(emd_dict)

# for param in emd.named_parameters():
#     if param[0] in encoder_keys or param[0] in decoder_keys:
#         param[1].requires_grad = False

lr = 1e-6
optimizer = optim.Adam(emd.parameters(), lr=lr)

recon_weight = 1.
kl_weight = 4.

torch.save(emd.state_dict(), MODELS_ROOT + 'EMD_' + 'prova' + '.pt')
exit()

train_unet_vae(net=emd, train_loader=train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, gmm=gmm,
               recon_weight=recon_weight, kl_weight=kl_weight, kl_annealing=kl_annealing, dataset='EMD', nn_type='conv',
               power=power)
