from models import PotentialMapperRaysNN
from constants import *
from utils import StrengthDataset, generate_dataset_from_strength
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import random

batch_size = 1
strengths = STRENGTHS

dataset = 'potential'
model_name = 'EMD_2021-05-23 09_20_56.202726.pt'
net_size = 1
conditional = False
mapping = False

model_path = MODELS_ROOT + model_name

pics_train_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_' + dataset + '.pt')
strength_train_dataset = torch.load(DATA_ROOT + 'num=999_unzipped/loaded_data/' + 'training_strength.pt')
pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)

train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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

h0 = POTENTIAL_LATENT_SIZE * 2
h1 = RAYS_LATENT_SIZE * 2

vae = PotentialMapperRaysNN(potential_image_channels=potential_image_channels,
                            rays_image_channels=rays_image_channels,
                            potential_hidden_size=potential_hidden_size,
                            rays_hidden_size=rays_hidden_size,
                            potential_latent_size=potential_latent_size,
                            rays_latent_size=rays_latent_size,
                            h_sizes=[h0, h1],
                            net_size=net_size)

vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

encodings = []
encoded_strenghts = []

max_samples = 1000

for i in range(max_samples):
    idx = random.randint(0, len(train_dataset) - 1)
    pic_sample = train_dataset[idx][0].unsqueeze(0).float()
    strength_sample = train_dataset[idx][1].unsqueeze(0)
    mean, log_var = vae.encode(pic_sample)
    x = torch.cat((mean, log_var), 1)
    if mapping:
        x = vae.mapping(x)
        mean = x[:, :rays_latent_size]
        log_var = x[:, rays_latent_size:]
    sample_encoded = torch.cat((mean, log_var), 0).flatten()
    
    sample_encoded = sample_encoded.tolist()

    encodings.append(sample_encoded)
    encoded_strenghts.append(train_dataset[idx][1])

encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

colors = [STRENGTHS_COLORS[round(strength.item(), 2)] for strength in encoded_strenghts]

fig, ax = plt.subplots()
scatter = ax.scatter(encodings_embedded[:, 0], encodings_embedded[:, 1], c=encoded_strenghts, cmap='plasma')
legend = ax.legend(*scatter.legend_elements(), loc="best", title="Strengths")
ax.add_artist(legend)
plt.title('T-SNE visualization of {} encodings'.format(dataset))
plt.show()
