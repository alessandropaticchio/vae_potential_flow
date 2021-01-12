from models import LinearVAE, ConvVAE
from training import train_vae, train_cvae
import torch
import torch.optim as optim

vae_type = 'lin'
dataset = 'MNIST'

if dataset == 'MNIST':
    from mnist_downloader import train_dataset, test_dataset
else:
    from fashion_mnist_downloader import train_dataset, test_dataset


batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if vae_type == 'conv':
    hidden_size = 3136
    vae = ConvVAE(image_dim=28, hidden_size=hidden_size, latent_size=int(hidden_size/2), image_channels=1)
else:
    vae = LinearVAE()
    
optimizer = optim.Adam(vae.parameters())

if torch.cuda.is_available():
    vae.cuda()

recon_weight = 1.
kl_weight = 1.

train_vae(net=vae, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer,
          recon_weight=recon_weight, kl_weight=kl_weight, dataset=dataset)
