from constants import *
from models import ConvVAE, ConvVAETest
from utils import generate_dataset_from_strength, StrengthDataset
import torch
import matplotlib.pyplot as plt
import random
import itertools

dataset = 'potential'
model_name = 'potential_VAE_[0.01, 0.3]_2021-07-09 08_38_32.842787.pt'
model_path = MODELS_ROOT + model_name
strengths = [0.01, 0.3]
train = True

pics_train_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_' + dataset + '.pt')
pics_test_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_' + dataset + '.pt')
strength_train_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'RP_images/loaded_data/' + 'test_strength.pt')
pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)
pics_test_dataset, strength_test_dataset = generate_dataset_from_strength(pics_test_dataset, strength_test_dataset,
                                                                          strengths)
train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)
test_dataset = StrengthDataset(x=pics_test_dataset, d=strength_test_dataset)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

if train:
    loader = train_loader
else:
    loader = test_loader

rand_sample_idx = max(0, random.randint(0, len(loader)) - 1)
rand_sample, rand_strength = next(itertools.islice(loader, rand_sample_idx, None))

if train:
    loader = train_loader
else:
    loader = test_loader

rand_sample_idx = max(0, random.randint(0, len(loader)) - 1)
rand_sample, rand_strength = next(itertools.islice(loader, rand_sample_idx, None))

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    # hidden_size = RAYS_HIDDEN_SIZE
    hidden_size = 8 * 47 * 47
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    # hidden_size = POTENTIAL_HIDDEN_SIZE
    hidden_size = 8 * 47 * 47
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

vae = ConvVAETest(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels)
vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

for layer in vae.named_parameters():
    name = layer[0]
    weights = layer[1].data.numpy()
    if name.startswith('conv') and 'bias' not in name:
        Tot_subplots = weights.shape[0]
        Cols = 4
        Rows = Tot_subplots // Cols
        Rows += Tot_subplots % Cols
        Position = range(1, Tot_subplots+1)

        fig = plt.figure(figsize=(15, 8))
        plt.title('Feature map {}'.format(name.split('.')[0]))
        feature_map = getattr(vae, name.split('.')[0])(rand_sample)[0].detach()
        rand_sample = feature_map.unsqueeze(0)
        for k in range(Tot_subplots):
            ax = fig.add_subplot(Rows, Cols, Position[k])
            ax.imshow(feature_map[k])

        plt.show()
