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
        mu = z[:, 0, :]  # the first feature values as mean
        log_var = z[:, 1, :]  # the other feature values as variance
        z = self.reparametrize(mu, log_var)
        x_prime = F.relu(self.dec1(z))
        x_prime = torch.sigmoid(self.dec2(x_prime))
        return x_prime


class Mapper(nn.Module):

    def __init__(self, h_sizes=[16, 16, 16]):
        super(Mapper, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            if i != len(self.hidden) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class PotentialConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(PotentialConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(image_channels, out_channels=32, kernel_size=4, stride=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=3)
        self.relu4 = nn.ReLU()

        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        self.deconv1 = nn.ConvTranspose2d(hidden_size, out_channels=128, kernel_size=6, stride=3)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(128, out_channels=64, kernel_size=4, stride=3)
        self.relu6 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(64, out_channels=32, kernel_size=4, stride=3)
        self.relu7 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(32, out_channels=image_channels, kernel_size=4, stride=3)
        self.relu8 = nn.ReLU()

        self.output = nn.Sigmoid()


    def reparametrize(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encode(x)

        #  Flattening
        x = x.view(x.size(0), -1)


        x, mean, log_var = self.decode(x)

        return x, mean, log_var

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        return x

    def decode(self, z):
        log_var = self.encoder_logvar(z)
        mean = self.encoder_mean(z)
        z = self.reparametrize(log_var, mean)
        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), self.hidden_size, 1, 1)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.deconv4(x)
        x = self.relu8(x)

        x_prime = self.output(x)

        return x_prime, mean, log_var


class RaysConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(RaysConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(image_channels, out_channels=32, kernel_size=4, stride=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=8, stride=3)
        self.relu4 = nn.ReLU()

        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        self.deconv1 = nn.ConvTranspose2d(hidden_size, out_channels=128, kernel_size=8, stride=3)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(128, out_channels=64, kernel_size=6, stride=3)
        self.relu6 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(64, out_channels=32, kernel_size=6, stride=3)
        self.relu7 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(32, out_channels=image_channels, kernel_size=6, stride=3)
        self.relu8 = nn.ReLU()

        self.output = nn.Sigmoid()


    def reparametrize(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encode(x)

        #  Flattening
        x = x.view(x.size(0), -1)

        x, mean, log_var = self.decode(x)

        return x, mean, log_var

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        return x

    def decode(self, z):
        log_var = self.encoder_logvar(z)
        mean = self.encoder_mean(z)
        z = self.reparametrize(log_var, mean)
        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), self.hidden_size, 1, 1)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.deconv4(x)
        x = self.relu8(x)

        x_prime = self.output(x)

        return x_prime, mean, log_var
