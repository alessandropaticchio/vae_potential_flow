from torch import nn
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

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=image_channels, kernel_size=3, stride=3, padding=20)

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

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.upsample1(x)

        x = self.conv6(x)

        x_prime = self.output(x)

        return x_prime


class DeConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(DeConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(image_channels, out_channels=32, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3)
        self.relu2 = nn.ReLU()

        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        self.deconv1 = nn.ConvTranspose2d(64, out_channels=32, kernel_size=1, stride=1)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv2 = nn.ConvTranspose2d(32, out_channels=16, kernel_size=3, stride=3)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.UpsamplingNearest2d(size=(20, 20))

        self.deconv3 = nn.ConvTranspose2d(16, out_channels=image_channels, kernel_size=3, stride=3, padding=20)
        self.relu7 = nn.ReLU()

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

        x = self.conv2(x)
        x = self.relu2(x)

        #  Flattening
        x = x.view(x.size(0), -1)

        mean = self.encoder_mean(x)
        log_var = self.encoder_logvar(x)

        return mean, log_var

    def decode(self, mean, log_var):
        z = self.reparametrize(mean=mean, log_var=log_var)
        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), 64, 3, 3)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.upsample1(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.upsample2(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x_prime = self.output(x)

        return x_prime
