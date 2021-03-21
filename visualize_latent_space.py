from models import ConvVAE
from constants import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

batch_size = 1
dataset = 'potential'

train_dataset = torch.load(DATA_ROOT + 'D=0.3 num=999/loaded_data/' + 'training_' + dataset + '.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model_name = 'potential_VAE__2021-03-21 14_31_19.807428.pt'
model_path = MODELS_ROOT + model_name

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
    hidden_size = RAYS_HIDDEN_SIZE
    latent_size = RAYS_LATENT_SIZE
    image_channels = RAYS_IMAGE_CHANNELS
else:
    image_size = POTENTIAL_IMAGE_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = POTENTIAL_LATENT_SIZE
    image_channels = POTENTIAL_IMAGE_CHANNELS

vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=image_channels,
              net_size=2)
vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

encodings = []

max_samples = 500

for i, idx in enumerate(range(max_samples)):
    sample = train_dataset[idx].unsqueeze(0).float()
    mean, log_var = vae.encode(sample)
    sample_encoded = torch.cat((mean, log_var), 0).flatten().tolist()
    encodings.append(sample_encoded)

encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

plt.figure()

plt.scatter(encodings_embedded[:, 0], encodings_embedded[:, 1])
plt.title('T-SNE visualization of {} encodings'.format(dataset))
plt.show()
