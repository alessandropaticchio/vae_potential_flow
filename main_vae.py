from fashion_mnist_downloader import train_dataset, test_dataset
from models import LinearVAE
from training import train_vae
import torch
import torch.optim as optim

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

vae = LinearVAE()
optimizer = optim.Adam(vae.parameters())

if torch.cuda.is_available():
    vae.cuda()

recon_weight = 1.
kl_weight = 150.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset='Fashion_MNIST')
