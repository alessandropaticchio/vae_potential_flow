from torch import nn
import torch


class ConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(ConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=3, stride=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=3)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv7 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.relu7 = nn.ReLU()

        self.upsample3 = nn.Upsample(scale_factor=2)

        self.conv8 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.relu8 = nn.ReLU()

        self.upsample4 = nn.Upsample(size=(100, 100))

        self.conv9 = nn.Conv2d(in_channels=4, out_channels=image_channels, kernel_size=1, stride=1, padding=0)
        self.relu9 = nn.ReLU()

        self.output = nn.Sigmoid()

    def reparametrize(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        mean, log_var = self.encode(x)

        x = self.decode(mean, log_var)

        return x, mean, log_var

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)

        # Flattening
        x = x.view(x.size(0), -1)

        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)

        return log_var, mean

    def decode(self, log_var, mean):
        z = self.reparametrize(log_var, mean)
        x = self.fc(z)

        # Unflattening
        x = x.view(x.size(0), 32, 5, 5)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.upsample1(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.upsample2(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.upsample3(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.upsample4(x)

        x = self.conv9(x)
        x = self.relu9(x)

        x_prime = self.output(x)

        return x_prime


class NewConvVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size, image_channels=3):
        super(NewConvVAE, self).__init__()
        self.image_channels = image_channels
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(image_channels, out_channels=32, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3)
        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=3)
        self.relu4 = nn.ReLU()

        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)

        self.deconv1 = nn.ConvTranspose2d(hidden_size, out_channels=128, kernel_size=1, stride=1)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(128, out_channels=64, kernel_size=1, stride=1)
        self.relu6 = nn.ReLU()

        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv3 = nn.ConvTranspose2d(64, out_channels=32, kernel_size=3, stride=3)
        self.relu7 = nn.ReLU()

        self.upsample2 = nn.UpsamplingNearest2d(size=(42, 42))

        self.deconv4 = nn.ConvTranspose2d(32, out_channels=image_channels, kernel_size=5, stride=3)
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

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.maxpool2(x)

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

        x = self.upsample1(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.upsample2(x)

        x = self.deconv4(x)
        x = self.relu8(x)

        x_prime = self.output(x)

        return x_prime, mean, log_var
