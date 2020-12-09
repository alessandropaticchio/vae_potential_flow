from mnist_downloader import train_dataset as mnist_train_dataset
from models import LinearVAE
from constants import *
import matplotlib.pyplot as plt
import torch

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)

model_name = 'MNIST_VAE_2020-12-07 09:50:15.947882.pt'
model_path = MODELS_ROOT + model_name

vae = LinearVAE()
vae.load_state_dict(torch.load(model_path))
vae.eval()

targets = range(0, 10)
mnist_train_targets = mnist_train_dataset.targets

encodings = []

max_samples = 500

for target in targets:
    mnist_train_indeces = (mnist_train_targets == target).nonzero().flatten().tolist()

    # Training set generation
    for i, idx in enumerate(mnist_train_indeces):
        mnist_sample = mnist_train_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample -= torch.min(mnist_sample)
        mnist_sample /= torch.max(mnist_sample)
        mnist_sample_encoded = vae.encode(mnist_sample).reshape(32).tolist()
        encodings.append(mnist_sample_encoded)

        if i == max_samples:
            break

from sklearn.manifold import TSNE

encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

plt.figure()

for i, target in enumerate(targets):
    plt.scatter(encodings_embedded[max_samples * i:max_samples * (i+1), 0], encodings_embedded[max_samples * i:max_samples * (i + 1), 1],
                label=target, cmap='Set1')
plt.title('T-SNE visualization of MNIST encodings')
plt.legend()
plt.show()
