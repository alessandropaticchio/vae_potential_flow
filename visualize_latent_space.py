from models import ConvVAE
from constants import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

batch_size = 1
dataset = 'rays'

train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model_name = 'rays_VAE_2020-12-21 11:16:48.097754.pt'
model_path = MODELS_ROOT + model_name

if dataset == 'rays':
    image_size = RAYS_IMAGE_SIZE
    hidden_size = RAYS_HIDDEN_SIZE
    latent_size = int(hidden_size / 2)
else:
    image_size = POTENTIAL_IMAGE_SIZE
    hidden_size = POTENTIAL_HIDDEN_SIZE
    latent_size = int(hidden_size / 2)

vae = ConvVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=1)
vae.load_state_dict(torch.load(model_path))
vae.eval()

encodings = []

max_samples = 500

for i, idx in enumerate(range(max_samples)):
    sample = train_dataset.X[idx].unsqueeze(0).float()
    mean, log_var = vae.encode(sample)
    sample_encoded = torch.cat((mean, log_var), 0).flatten().tolist()
    encodings.append(sample_encoded)


encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

plt.figure()

plt.scatter(encodings_embedded[:, 0], encodings_embedded[:, 1])
plt.title('T-SNE visualization of {} encodings'.format(dataset))
plt.show()
