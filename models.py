from torch import nn
import torch
import torch.nn.functional as F


class DenseVAE(nn.Module):
    def __init__(self, features=16, in_features=784, out_features=512):
        super(DenseVAE, self).__init__()
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


class ConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(ConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # encode
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # latent space
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=3)

        self.output = nn.Sigmoid()

    def reparametrize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)

        x = self.decode(mean=mean, log_var=log_var)

        return x, mean, log_var

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)

        # Flattening
        x = x.view(x.size(0), -1)

        mean = self.encoder_mean(x)
        log_var = self.encoder_logvar(x)

        return mean, log_var

    def decode(self, mean, log_var):
        z = self.reparametrize(mean=mean, log_var=log_var)
        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), 16, 12, 12)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = self.relu3(x)

        x = self.deconv2(x)

        x_prime = self.output(x)

        return x_prime


# define a Conv VAE

class ConvVAETest(nn.Module):

    def __init__(self):
        super(ConvVAETest, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_mu = nn.Linear(12 * 12 * 16, 20)
        self.fc1_sig = nn.Linear(12 * 12 * 16, 20)
        self.fc2 = nn.Linear(20, 12 * 12 * 16)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.ConvTranspose2d(16, 32, 3)
        self.conv4 = nn.ConvTranspose2d(32, 1, 3)

    def encode(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(a1))
        mx_poold = self.max_pool(a2)
        a_reshaped = mx_poold.reshape(-1, 12 * 12 * 16)
        a_mu = self.fc1_mu(a_reshaped)
        a_logvar = self.fc1_sig(a_reshaped)
        return a_mu, a_logvar

    def decode(self, mean, log_var):
        z = self.reparameterize(mean, log_var)

        a3 = F.relu(self.fc2(z))
        a3 = a3.reshape(-1, 16, 12, 12)
        a3_upsample = self.up_sample(a3)
        a4 = F.relu(self.conv3(a3_upsample))
        a5 = torch.sigmoid(self.conv4(a4))
        return a5

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mean=mu, log_var=logvar), mu, logvar

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


class ConvMapper(nn.Module):

    def __init__(self, mnist_encoded_size, fashion_mnist_encoded_size):
        super(ConvMapper, self).__init__()
        scale_factor = fashion_mnist_encoded_size / mnist_encoded_size

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=scale_factor)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.upsample(x)

        x = self.conv4(x)

        return x
