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
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3, net_size=1, conditional=False):
        super(ConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size * net_size
        self.latent_size = latent_size
        self.net_size = net_size
        self.conditional = conditional

        # encode
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32 * net_size, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(num_features=32 * net_size)

        self.conv2 = nn.Conv2d(in_channels=32 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        if conditional:
            # latent space, + 1 is for the strength
            reduced_hidden_size = self.hidden_size // 1000
            self.conditional_layer = nn.Linear(self.hidden_size + 1, reduced_hidden_size)
            self.encoder_mean = nn.Linear(reduced_hidden_size, self.latent_size)
            self.encoder_logvar = nn.Linear(reduced_hidden_size, self.latent_size)
            self.fc = nn.Linear(self.latent_size + 1, self.hidden_size)
        else:
            self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
            self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
            self.fc = nn.Linear(self.latent_size, self.hidden_size)

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm2d(num_features=32 * net_size)

        self.deconv2 = nn.ConvTranspose2d(in_channels=32 * net_size, out_channels=image_channels, kernel_size=3)

        self.output = nn.Sigmoid()

    def reparametrize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, strength):
        mean, log_var = self.encode(x=x, strength=strength)

        x = self.decode(mean=mean, log_var=log_var, strength=strength)

        return x, mean, log_var

    def encode(self, x, strength):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)

        # Flattening
        x = x.view(x.size(0), -1)

        if self.conditional:
            #  Concatenating strength
            x = torch.cat([x, strength], dim=1)
            x = self.conditional_layer(x)

        mean = self.encoder_mean(x)
        log_var = self.encoder_logvar(x)

        return mean, log_var

    def decode(self, mean, log_var, strength):
        z = self.reparametrize(mean=mean, log_var=log_var)

        if self.conditional:
            #  Concatenating strength
            z = torch.cat([z, strength], dim=1)

        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), 16 * self.net_size, 48, 48)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = self.relu3(x)
        x = self.batch_norm2(x)

        x = self.deconv2(x)

        x_prime = self.output(x)

        return x_prime


class ConvVAETest(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3, net_size=1, conditional=False):
        super(ConvVAETest, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size * net_size
        self.latent_size = latent_size
        self.net_size = net_size
        self.conditional = conditional

        # encode
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64 * net_size, kernel_size=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=64 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=32 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels=16 * net_size, out_channels=8 * net_size, kernel_size=3)
        self.relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(in_channels=8 * net_size, out_channels=4 * net_size, kernel_size=3)
        self.relu5 = nn.LeakyReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        if conditional:
            # latent space, + 1 is for the strength
            reduced_hidden_size = self.hidden_size // 1000
            self.conditional_layer = nn.Linear(self.hidden_size + 1, reduced_hidden_size)
            self.encoder_mean = nn.Linear(reduced_hidden_size, self.latent_size)
            self.encoder_logvar = nn.Linear(reduced_hidden_size, self.latent_size)
            self.fc = nn.Linear(self.latent_size + 1, self.hidden_size)
        else:
            self.encoder_mean = nn.Linear(self.hidden_size, self.latent_size)
            self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)
            self.fc = nn.Linear(self.latent_size, self.hidden_size)

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=4 * net_size, out_channels=8 * net_size, kernel_size=1)
        self.relu5 = nn.LeakyReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=8 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu6 = nn.LeakyReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=16 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu7 = nn.LeakyReLU()

        self.deconv4 = nn.ConvTranspose2d(in_channels=32 * net_size, out_channels=64 * net_size, kernel_size=3)
        self.relu8 = nn.LeakyReLU()

        self.deconv5 = nn.ConvTranspose2d(in_channels=64 * net_size, out_channels=image_channels, kernel_size=3)

        self.output = nn.Sigmoid()

    def reparametrize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, strength):
        mean, log_var = self.encode(x=x, strength=strength)

        x = self.decode(mean=mean, log_var=log_var, strength=strength)

        return x, mean, log_var

    def encode(self, x, strength):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.maxpool1(x)

        # Flattening
        x = x.view(x.size(0), -1)

        if self.conditional:
            #  Concatenating strength
            x = torch.cat([x, strength], dim=1)
            x = self.conditional_layer(x)

        mean = self.encoder_mean(x)
        log_var = self.encoder_logvar(x)

        return mean, log_var

    def decode(self, mean, log_var, strength):
        z = self.reparametrize(mean=mean, log_var=log_var)

        if self.conditional:
            #  Concatenating strength
            z = torch.cat([z, strength], dim=1)

        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), 4 * self.net_size, 46, 46)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.deconv4(x)
        x = self.relu8(x)

        x = self.deconv5(x)

        x_prime = self.output(x)
        # x_prime = 0.01 / (1 + torch.exp((-0.01 * x))) - 0.01/2

        return x_prime


class Mapper(nn.Module):

    def __init__(self, h_sizes=[16, 16, 16]):
        super(Mapper, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.layers.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
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
        self.conv1 = nn.Conv2d(in_channels=potential_image_channels, out_channels=32 * net_size, kernel_size=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16 * net_size, out_channels=8 * net_size, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=8 * net_size, out_channels=4 * net_size, kernel_size=3)
        self.relu4 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(self.potential_hidden_size, self.potential_latent_size)
        self.encoder_logvar = nn.Linear(self.potential_hidden_size, self.potential_latent_size)
        self.fc = nn.Linear(self.potential_latent_size, self.potential_hidden_size)

        # Mapper
        self.mapper_layers = nn.ModuleList()

        for k in range(len(h_sizes) - 1):
            self.mapper_layers.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=4 * net_size, out_channels=8 * net_size, kernel_size=1)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=8 * net_size, out_channels=16 * net_size, kernel_size=3)
        self.relu6 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=16 * net_size, out_channels=32 * net_size, kernel_size=3)
        self.relu7 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(in_channels=32 * net_size, out_channels=rays_image_channels, kernel_size=3)

        self.output = nn.Sigmoid()

    def forward(self, x):
        potential_mean, potential_log_var = self.encode(x)

        x = torch.cat((potential_mean, potential_log_var), 1)

        x = self.mapping(x)

        rays_mean = x[:, :self.rays_latent_size]
        rays_log_var = x[:, self.rays_latent_size:]

        x_prime = self.decode(mean=rays_mean, log_var=rays_log_var)

        return x_prime, rays_mean, rays_log_var

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
        x = x.view(x.size(0), 4 * self.net_size, 47, 47)

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

    def mapping(self, x):
        for i, layer in enumerate(self.mapper_layers):
            if i != len(self.mapper_layers) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x

    def reparametrize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


class ConvPlainAE(nn.Module):
    def __init__(self, image_channels=3, net_size=1):
        super(ConvPlainAE, self).__init__()
        self.image_channels = image_channels
        self.net_size = net_size

        # encode
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=4 * net_size, kernel_size=1)
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.Identity()
        self.relu1 = nn.LeakyReLU()
        # self.relu1 = nn.Sigmoid()

        self.conv2 = nn.Conv2d(in_channels=4 * net_size, out_channels=2 * net_size, kernel_size=3)
        # self.relu2 = nn.ReLU()
        # self.relu2 = nn.Identity()
        self.relu2 = nn.LeakyReLU()
        # self.relu2 = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(in_channels=2 * net_size, out_channels=4 * net_size, kernel_size=3)
        # self.relu3 = nn.ReLU()
        # self.relu3 = nn.Identity()
        self.relu3 = nn.LeakyReLU()
        # self.relu3 = nn.Sigmoid()

        self.deconv2 = nn.ConvTranspose2d(in_channels=4 * net_size, out_channels=image_channels, kernel_size=1)

        self.output = nn.Sigmoid()
        # self.output = nn.ReLU()
        # self.output = nn.Identity()
        # self.output = nn.LeakyReLU()

    def forward(self, x):
        x = self.encode(x=x)

        x = self.decode(x=x)

        return x

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.relu3(x)

        x = self.deconv2(x)

        x_prime = self.output(x)

        return x_prime


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.uniform_(-30., 30.)
