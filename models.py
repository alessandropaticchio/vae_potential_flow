from torch import nn
import torch.nn.functional as F
import torch


class ConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(ConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=image_channels, kernel_size=3, stride=1, padding=1)

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
        x = x.view(x.size(0), 32, 10, 10)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.upsample1(x)

        x = self.conv3(x)

        x_prime = self.output(x)

        return x_prime


class ConvPlainAE(nn.Module):
    def __init__(self, image_dim, image_channels=3):
        super(ConvPlainAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=8, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=image_channels, kernel_size=3, stride=1, padding=1)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.encode(x)

        x_prime = self.decode(x)

        return x_prime

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.maxpool1(x)

        return x

    def decode(self, x):
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.upsample1(x)

        x = self.conv3(x)

        x_prime = self.output(x)

        return x_prime


class Mapper(nn.Module):

    def __init__(self, h_sizes):
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

    def __init__(self, potential_encoded_size, rays_encoded_size):
        super(ConvMapper, self).__init__()
        scale_factor = rays_encoded_size / potential_encoded_size

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=scale_factor)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0)

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
