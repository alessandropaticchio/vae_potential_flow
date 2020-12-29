import pytorch_lightning as pl
pl.seed_everything(1234)
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
from pl_bolts.datamodules import CIFAR10DataModule
from image_plotting_callback import ImageSampler
from argparse import ArgumentParser
from constants import *



class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
        # return torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        # x, _ = batch
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        dataset_test = CIFAR10DataModule('.')
    if args.dataset == 'imagenet':
        dataset_test = ImagenetDataModule('.')

    batch_size = 8

    dataset = 'potential'
    train_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/training_' + dataset + '.pt')
    test_dataset = torch.load(DATA_ROOT + 'real_data/' + dataset + '_pic_data/test_' + dataset + '.pt')

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset[1, :, :, :].unsqueeze(0), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if dataset == 'rays':
        image_size = RAYS_IMAGE_SIZE
        hidden_size = RAYS_HIDDEN_SIZE
        latent_size = int(hidden_size / 2)
        image_channels = RAYS_IMAGE_CHANNELS
    else:
        image_size = POTENTIAL_IMAGE_SIZE
        hidden_size = POTENTIAL_HIDDEN_SIZE
        latent_size = int(hidden_size / 2)
        image_channels = POTENTIAL_IMAGE_CHANNELS

    sampler = ImageSampler()

    vae = VAE(input_height=image_size)
    # trainer = pl.Trainer(gpus=args.gpus, max_epochs=20, callbacks=[sampler])
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=100)
    trainer.fit(vae, train_loader)


if __name__ == '__main__':
    train()