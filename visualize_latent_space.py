from models import ConvVAE
from constants import *
from utils import StrengthDataset, generate_dataset_from_strength
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

batch_size = 1
strengths = STRENGTHS[:5]
dataset = 'rays'
model_name = 'rays_VAE_[0.01, 0.1, 0.2, 0.03, 0.3]_2021-04-18 12_27_15.732017.pt'
conditional = True
model_path = MODELS_ROOT + model_name

pics_train_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_' + dataset + '.pt')
strength_train_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_strength.pt')
pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)

train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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
              net_size=1, conditional=conditional)
vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vae.eval()

encodings = []
encoded_strenghts = []

max_samples = 999

for i, idx in enumerate(range(max_samples)):
    pic_sample = train_dataset[idx][0].unsqueeze(0).float()
    strength_sample = train_dataset[idx][1].unsqueeze(0)
    mean, log_var = vae.encode(pic_sample, strength_sample)
    sample_encoded = torch.cat((mean, log_var), 0).flatten().tolist()
    encodings.append(sample_encoded)
    encoded_strenghts.append(train_dataset[idx][1])

encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

fig, ax = plt.subplots()
scatter = ax.scatter(encodings_embedded[:, 0], encodings_embedded[:, 1], c=encoded_strenghts, cmap='Accent')
legend = ax.legend(*scatter.legend_elements(),
                   loc="best", title="Strengths")
ax.add_artist(legend)
plt.title('T-SNE visualization of {} encodings'.format(dataset))
plt.show()
