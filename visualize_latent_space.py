from models import DenseVAE, ConvVAE
from constants import *
import matplotlib.pyplot as plt
import torch

vae_type = 'conv'
dataset = 'MNIST'
subset = True

if subset:
    train_dataset = torch.load(DATA_ROOT + 'subsets/' + dataset + '/training.pt')
else:
    from mnist_downloader import train_dataset as train_dataset

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model_name = 'Fashion_MNIST_VAE_2021-01-17 15:25:48.940916.pt'
model_path = MODELS_ROOT + model_name

if vae_type == 'conv':
    vae = ConvVAE(image_dim=28, hidden_size=HIDDEN_SIZE, latent_size=int(HIDDEN_SIZE / 2), image_channels=1)
else:
    vae = DenseVAE()
vae.load_state_dict(torch.load(model_path))
vae.eval()

targets = range(0, 10)
train_targets = train_dataset.targets

encodings = []

sample_class_size = 100

for target in targets:
    if subset:
        train_indeces = (train_targets.squeeze() == target).nonzero().flatten().tolist()
    # Training set generation
    for i, idx in enumerate(train_indeces):
        if subset:
            sample = train_dataset.data[idx].unsqueeze(0).float()
        else:
            sample = train_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        sample -= torch.min(sample)
        sample /= torch.max(sample)
        if subset:
            sample_encoded = vae.encode(sample)[0].flatten().tolist()
        else:
            sample_encoded = vae.encode(sample).reshape(32).tolist()
        encodings.append(sample_encoded)

        if i == sample_class_size:
            break

from sklearn.manifold import TSNE

encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

plt.figure()

for i, target in enumerate(targets):
    plt.scatter(encodings_embedded[sample_class_size * i:sample_class_size * (i + 1), 0], encodings_embedded[sample_class_size * i:sample_class_size * (i + 1), 1],
                label=target, cmap='Set1')
plt.title('T-SNE visualization of ' + dataset + ' encodings')
plt.legend()
plt.show()
