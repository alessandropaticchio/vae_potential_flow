from torch import nn
import torch
import torch.nn.functional as F


class DenseVAE(nn.Module):
    def __init__(self, out_features, features=16, in_features=784):
        super(DenseVAE, self).__init__()
        self.features = features

        # encoder
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=out_features, out_features=self.features * 2)

        # decoder
        self.fc3 = nn.Linear(in_features=self.features, out_features=out_features)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=out_features, out_features=in_features)

        self.output = nn.Sigmoid()

    def reparametrize(self, mean, log_var):
        """
        :param mean: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mean + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        mean, log_var, x_to_skip = self.encode(x)

        x_prime = self.decode(mean=mean, log_var=log_var, to_skip=x_to_skip)

        return x_prime

    def encode(self, x):
        # Flattening
        x = x.view(-1, x.size(2) * x.size(3))

        # encoding
        x = self.fc1(x)
        enc1 = self.relu1(x)
        x = self.fc2(enc1).view(-1, 2, self.features)

        # get `mu` and `log_var`
        mean = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        return mean, log_var, enc1

    def decode(self, mean, log_var, enc1):
        # get the latent vector through reparametrization
        z = self.reparametrize(mean, log_var)

        # decoding
        x_prime = self.fc3(z)
        x_prime = x_prime + enc1
        x_prime = self.relu2(x_prime)
        x_prime = self.fc4(x_prime)

        x_prime = self.output(x_prime)
        return x_prime, mean, log_var


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


class DeConvVAE(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(DeConvVAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.encoder_log_var = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.fc = nn.Linear(in_features=self.latent_size, out_features=self.hidden_size)
        self.relu3 = nn.ReLU()

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu4 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3)

        self.output = nn.Sigmoid()

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.max_pool(x)
        x = x.reshape(-1, self.hidden_size)

        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        return mean, log_var

    def decode(self, mean, log_var):
        z = self.reparameterize(mean, log_var)

        x = self.fc(z)
        x = self.relu3(x)

        x = x.reshape(-1, 16, 12, 12)

        x = self.up_sample(x)

        x = self.conv3(x)
        x = self.relu4(x)

        x = self.conv4(x)
        x_prime = self.output(x)

        return x_prime

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mean=mu, log_var=logvar), mu, logvar


class SkipDeConvVAE(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(SkipDeConvVAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.encoder_log_var = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.fc = nn.Linear(in_features=self.latent_size, out_features=self.hidden_size)
        self.relu3 = nn.ReLU()

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu4 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3)

        self.output = nn.Sigmoid()

    def encode(self, x):
        x = self.conv1(x)
        enc1 = self.relu1(x)

        x = self.conv2(enc1)
        enc2 = self.relu2(x)

        x = self.max_pool(enc2)
        x = x.reshape(-1, self.hidden_size)

        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        return mean, log_var, enc1, enc2

    def decode(self, mean, log_var, enc1, enc2):
        z = self.reparameterize(mean, log_var)

        x = self.fc(z)
        x = self.relu3(x)

        x = x.reshape(-1, 32, 12, 12)

        x = self.up_sample(x)

        #  Skip connection
        x = x + enc2

        x = self.deconv1(x)
        #  Skip connection
        x = x + enc1
        x = self.relu4(x)

        x = self.deconv2(x)
        x_prime = self.output(x)

        return x_prime

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar, enc1, enc2 = self.encode(x)
        return self.decode(mean=mu, log_var=logvar, enc1=enc1, enc2=enc2), mu, logvar


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


class UNet_VAE(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(UNet_VAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.encoder_log_var = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.fc = nn.Linear(in_features=self.latent_size, out_features=self.hidden_size)
        self.relu4 = nn.ReLU()

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=3)
        self.relu5 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3)

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

        x = x.reshape(-1, 32, 11, 11)

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


class Mini_UNet_VAE(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(Mini_UNet_VAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.encoder_log_var = nn.Linear(in_features=self.hidden_size, out_features=self.latent_size)
        self.fc = nn.Linear(in_features=self.latent_size, out_features=self.hidden_size)
        self.relu4 = nn.ReLU()

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3)

        self.output = nn.Sigmoid()

    def encode(self, x):
        x = self.conv1(x)
        enc1 = self.relu1(x)

        enc1_pool = self.max_pool(enc1)

        enc1_pool = enc1_pool.reshape(-1, self.hidden_size)

        mean = self.encoder_mean(enc1_pool)
        log_var = self.encoder_log_var(enc1_pool)
        return mean, log_var, enc1

    def decode(self, mean, log_var, enc1):
        z = self.reparameterize(mean, log_var)

        x = self.fc(z)
        x = self.relu4(x)

        x = x.reshape(-1, 32, 13, 13)

        x = self.up_sample(x)

        x = torch.cat((x, enc1), dim=1)

        x = self.deconv1(x)

        x_prime = self.output(x)

        return x_prime

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar, enc1 = self.encode(x)
        return self.decode(mean=mu, log_var=logvar, enc1=enc1), mu, logvar
