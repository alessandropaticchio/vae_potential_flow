from models import DenseVAE, UNet_VAE
from constants import *
from utils import MyDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

vae_type = 'conv'
dataset = 'Total'
subset = True

if subset:
    if dataset == 'Total':
        if dataset == 'Total':
            mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/' + 'MNIST' + '/training.pt')
            mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/' + 'MNIST' + '/test.pt')
            fashion_mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/' + 'Fashion_MNIST' + '/training.pt')
            fashion_mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/' + 'Fashion_MNIST' + '/test.pt')
            train_dataset = MyDataset(x=mnist_train_dataset.data, y=fashion_mnist_train_dataset.data)
            test_dataset = MyDataset(x=mnist_test_dataset.data, y=fashion_mnist_test_dataset.data)
    else:
        train_dataset = torch.load(DATA_ROOT + 'subsets/' + dataset + '/training.pt')

else:
    from mnist_downloader import train_dataset as train_dataset

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model_name = 'Total_VAE__2021-02-04 11:05:04.531770.pt'
model_path = MODELS_ROOT + model_name

if vae_type == 'conv':
    hidden_size = 32 * 11 * 11
    vae = UNet_VAE(hidden_size=hidden_size, latent_size=LATENT_SIZE)
else:
    vae = DenseVAE()
vae.load_state_dict(torch.load(model_path))
vae.eval()

encodings = []
targets = range(0, 10)
sample_class_size = 100

if dataset != 'Total':
    train_targets = train_dataset.targets
    for target in targets:
        if subset:
            train_indeces = (train_targets.squeeze() == target).nonzero().flatten().tolist()
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

else:
    train_targets = mnist_train_dataset.targets
    for i, target in enumerate(targets):
        train_indeces = (train_targets.squeeze() == target).nonzero().flatten().tolist()
        for i, idx in enumerate(train_indeces):
            sample = mnist_train_dataset.data.data[idx].unsqueeze(0).float()
            sample -= torch.min(sample)
            sample /= torch.max(sample)
            sample_encoded = vae.encode(sample)[0].flatten().tolist()
            encodings.append(sample_encoded)

            if i == sample_class_size:
                break

encodings_embedded = TSNE(n_components=2).fit_transform(encodings)

plt.figure()

for i, target in enumerate(targets):
    plt.scatter(encodings_embedded[sample_class_size * i:sample_class_size * (i + 1), 0],
                encodings_embedded[sample_class_size * i:sample_class_size * (i + 1), 1],
                label=target, cmap='Set1')
    plt.title('T-SNE visualization of ' + dataset + ' encodings')

plt.legend()
plt.show()

