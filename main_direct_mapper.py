from models import *
from training import train_total_vae
from constants import *
from utils import MyDataset
import torch
import torch.optim as optim

vae_type = 'conv'
dataset = 'Total'
subset = True

if subset:
    mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/' + 'MNIST' + '/training.pt')
    mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/' + 'MNIST' + '/test.pt')
    fashion_mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/' + 'Fashion_MNIST' + '/training.pt')
    fashion_mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/' + 'Fashion_MNIST' + '/test.pt')
else:
    raise NotImplementedError()

# Compose datasets where X are mnist samples, y are fashion_mnist samples. The matching before samples from the same
# class is respected due to the generation of the subsets. TODO: Has to be fixed when going to full dataset
train_dataset = MyDataset(x=mnist_train_dataset.data, y=fashion_mnist_train_dataset.data)
test_dataset = MyDataset(x=mnist_test_dataset.data, y=fashion_mnist_test_dataset.data)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

hidden_size = 16 * 12 * 12
vae = DeConvVAETest(hidden_size=hidden_size, latent_size=LATENT_SIZE)

optimizer = optim.Adam(vae.parameters())

if torch.cuda.is_available():
    vae.cuda()

recon_weight = 1.
kl_weight = 5.

train_total_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=500, optimizer=optimizer,
                recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset, nn_type=vae_type)
