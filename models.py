from torch import nn
import torch
import torch.nn.functional as F


class UNet_VAE(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(UNet_VAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.encoder_log_var = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.fc = nn.Linear(in_features=self.latent_size, out_features=self.hidden_size)
        self.relu4 = nn.ReLU()

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3)
        self.relu5 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3)

        self.output = nn.Sigmoid()

    def encode(self, x):
        x = self.conv1(x)
        enc1 = self.relu1(x)

        x = self.conv2(enc1)
        enc2 = self.relu2(x)

        x = self.conv3(enc2)
        enc3 = self.relu3(x)

        enc3_pool = self.max_pool(enc3)

        enc3_pool = enc3_pool.reshape(-1, self.hidden_size)

        mean = self.encoder_mean(enc3_pool)
        log_var = self.encoder_log_var(enc3_pool)
        return mean, log_var, enc1, enc2, enc3

    def decode(self, mean, log_var, enc1, enc2, enc3):
        z = self.reparameterize(mean, log_var)

        x = self.fc(z)
        x = self.relu4(x)

        x = x.reshape(-1, 128, 147, 147)

        x = self.up_sample(x)

        x = torch.cat((x, enc3), dim=1)

        x = self.deconv1(x)
        x = self.relu4(x)

        x = torch.cat((x, enc2), dim=1)

        x = self.deconv2(x)
        x = self.relu5(x)

        x = torch.cat((x, enc1), dim=1)

        x = self.deconv3(x)

        x_prime = self.output(x)

        return x_prime

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar, enc1, enc2, enc3 = self.encode(x)
        return self.decode(mean=mu, log_var=logvar, enc1=enc1, enc2=enc2, enc3=enc3), mu, logvar


class ConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3, net_size=1):
        super(ConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size * net_size
        self.latent_size = latent_size
        self.net_size = net_size

        # encode
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32 * net_size, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # latent space
        self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
        self.fc = nn.Linear(self.latent_size, self.hidden_size)

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=32 * net_size, out_channels=image_channels, kernel_size=3)

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
        x = x.view(x.size(0), 16 * self.net_size, 148, 148)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = self.relu3(x)

        x = self.deconv2(x)

        x_prime = self.output(x)

        return x_prime


class ConvVAETest(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3, net_size=1):
        super(ConvVAETest, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size * net_size
        self.latent_size = latent_size
        self.net_size = net_size

        # encode
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32 * net_size, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16 * net_size, out_channels=8 * net_size, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=8 * net_size, out_channels=4 * net_size, kernel_size=3)
        self.relu4 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # latent space
        self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
        self.fc = nn.Linear(self.latent_size, self.hidden_size)

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=4 * net_size, out_channels=8 * net_size, kernel_size=3)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=8 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu6 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=16 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu7 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(in_channels=32 * net_size, out_channels=image_channels, kernel_size=3)

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

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

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
        x = x.view(x.size(0), 4 * self.net_size, 146, 146)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.deconv4(x)

        x_prime = self.output(x)

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


class PotentialMapperRaysNN(nn.Module):

    def __init__(self, potential_image_channels, rays_image_channels, potential_hidden_size, rays_hidden_size,
                 potential_latent_size, rays_latent_size, net_size, h_sizes):
        super(PotentialMapperRaysNN, self).__init__()
        self.potential_image_channels = potential_image_channels
        self.rays_image_channels = rays_image_channels
        self.potential_hidden_size = potential_hidden_size
        self.rays_hidden_size = rays_hidden_size
        self.potential_latent_size = potential_latent_size
        self.rays_latent_size = rays_latent_size
        self.net_size = net_size

        # encode
        self.conv1 = nn.Conv2d(in_channels=potential_image_channels, out_channels=32 * net_size, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # latent space
        self.encoder_mean = nn.Linear(self.potential_hidden_size, self.potentiallatent_size)
        self.encoder_logvar = nn.Linear(self.potential_hidden_size, self.potential_latent_size)
        self.fc = nn.Linear(self.potential_latent_size, self.potential_hidden_size)

        # Mapper
        self.hidden = nn.ModuleList()

        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=32 * net_size, out_channels=rays_image_channels, kernel_size=3)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.encode(x)

        x = self.mapper(x)

        mean = x[:, :self.rays_latent_size]
        log_var = x[:, self.rays_latent_size:]

        x_prime = self.decode(mean=mean, log_var=log_var)

        return x_prime

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
        x = x.view(x.size(0), 16 * self.net_size, 148, 148)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = self.relu3(x)

        x = self.deconv2(x)

        x_prime = self.output(x)

        return x_prime

    def mapper(self, x):
        for i, layer in enumerate(self.hidden):
            if i != len(self.hidden) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x
