from constants import *
from datetime import datetime
from torch.nn import MSELoss
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def train_mapper(net, train_loader, test_loader, epochs, optimizer):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format('Mapper_' + now))
    net = net.to(device)
    net.train()
    mse_loss = MSELoss()
    for epoch in range(epochs):
        train_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = net(data)
            loss = mse_loss(output, target)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        test_loss = test_mapper(net, test_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + 'Mapper_' + now + '.pt')


def test_mapper(net, test_loader):
    net.eval()
    net = net.to(device)
    test_loss = 0
    mse_loss = MSELoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)

            test_loss += mse_loss(output, target).item()

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def train_vae(net, train_loader, test_loader, epochs, optimizer, recon_weight=1., kl_weight=1., dataset='MNIST'):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format(dataset + '_VAE_' + now))
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        bce_loss = 0.
        kld_loss = 0.
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = net(data)
            batch_loss, batch_bce_loss, batch_kld_loss = loss_function_vae(recon_batch, data, mu, log_var, recon_weight,
                                                                           kl_weight)

            batch_loss.backward()
            train_loss += batch_loss.item()
            bce_loss += batch_bce_loss.item()
            kld_loss += batch_kld_loss.item()

            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), batch_loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        test_loss = test_vae(net, test_loader, recon_weight, kl_weight)

        writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/bce_train', bce_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/kld_train', kld_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/test', test_loss / len(test_loader.dataset), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + now + '.pt')


def test_vae(net, test_loader, recon_weight, kl_weight):
    net.eval()
    net = net.to(device)
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = net(data)

            # sum up batch loss
            test_loss += loss_function_vae(recon, data, mu, log_var, recon_weight, kl_weight)[0].item()

    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


# return reconstruction error + KL divergence losses
def loss_function_vae(recon_x, x, mu, log_var, recon_weight, kl_weight):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') * recon_weight
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') * recon_weight
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * kl_weight
    return BCE + KLD, BCE, KLD
