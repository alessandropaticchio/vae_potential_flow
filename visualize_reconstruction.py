from models import *
from constants import *
from utils import MyDataset
import matplotlib.pyplot as plt
import random
import torch
import itertools

vae_type = 'conv'
dataset = 'Total'
subset = True
model_name = 'Total_VAE__2021-01-30 11:16:05.504872.pt'
model_path = MODELS_ROOT + model_name

if subset:
    if dataset == 'Total':
        mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/' + 'MNIST' + '/training.pt')
        mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/' + 'MNIST' + '/test.pt')
        fashion_mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/' + 'Fashion_MNIST' + '/training.pt')
        fashion_mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/' + 'Fashion_MNIST' + '/test.pt')
        train_dataset = MyDataset(x=mnist_train_dataset.data, y=fashion_mnist_train_dataset.data)
        test_dataset = MyDataset(x=mnist_test_dataset.data, y=fashion_mnist_test_dataset.data)
    else:
        train_dataset = torch.load(DATA_ROOT + 'subsets/' + dataset + '/training.pt')
        test_dataset = torch.load(DATA_ROOT + 'subsets/' + dataset + '/test.pt')
else:
    if dataset == 'MNIST':
        from mnist_downloader import train_dataset, test_dataset
    elif dataset == 'Fashion_MNIST':
        from fashion_mnist_downloader import train_dataset, test_dataset
    else:
        raise NotImplementedError

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if vae_type == 'conv':
    hidden_size = 16 * 11 * 11
    vae = DeConvVAETest(hidden_size=hidden_size, latent_size=LATENT_SIZE)
else:
    vae = DenseVAE(out_features=100)

vae.load_state_dict(torch.load(model_path))
vae.eval()

rand_sample_idx = random.randint(0, 1000)
rand_sample = next(itertools.islice(test_loader, rand_sample_idx, None))

rand_sample_prime = vae(rand_sample[0])[0]


plt.figure()
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(rand_sample[0].reshape(28, 28).detach().numpy())

plt.subplot(1, 2, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.reshape(28, 28).detach().numpy())

plt.show()
