from torch import nn
import torch
import torch.nn.functional as F


class LinearVAE(nn.Module):
    def __init__(self, features=16, in_features=784, out_features=512):
        super(LinearVAE, self).__init__()
        self.features = features

        # encoder
        self.enc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.enc2 = nn.Linear(in_features=out_features, out_features=features * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=features, out_features=out_features)
        self.dec2 = nn.Linear(in_features=out_features, out_features=in_features)

    def reparametrize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # Flattening
        x = x.view(-1, x.size(2) * x.size(3))

        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparametrization
        z = self.reparametrize(mu, log_var)

        # decoding
        x_prime = F.relu(self.dec1(z))
        x_prime = torch.sigmoid(self.dec2(x_prime))
        return x_prime, mu, log_var

    def encode(self, x):
        # Flattening
        x = x.view(-1, x.size(2) * x.size(3))

        # encoding
        x = F.relu(self.enc1(x))
        z = self.enc2(x).view(-1, 2, self.features)
        return z

    def decode(self, z):
        # decoding
        x_prime = F.relu(self.dec1(z))
        x_prime = torch.sigmoid(self.dec2(x_prime))
        return x_prime


class Mapper(nn.Module):

    def __init__(self, h_sizes=[16, 16, 16]):
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

    def forward(self, x):
        for layer in self.hidden():
            x = F.relu(layer(x))
        return x
