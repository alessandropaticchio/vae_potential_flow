from constants import *
from datetime import datetime
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import numpy as np


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

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))
        test_loss = test_mapper(net, test_loader)

        writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/test', test_loss / len(train_loader.dataset), epoch)

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

    print('====> Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))
    return test_loss


def train_vae(net, train_loader, test_loader, epochs, optimizer, recon_weight=1., kl_weight=1., dataset='MNIST',
              nn_type='conv', desc=''):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format(dataset + '_VAE_' + desc + '_' + now))
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_recon_loss = 0.
        train_kld_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = net(data)
            batch_loss, batch_recon_loss, batch_kld_loss = loss_function_vae(recon_batch, data, mu, log_var,
                                                                             recon_weight,
                                                                             kl_weight, nn_type)

            batch_loss.backward()
            train_loss += batch_loss.item()
            train_recon_loss += batch_recon_loss.item()
            train_kld_loss += batch_kld_loss.item()

            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

        test_loss, test_recon_loss, test_kld_loss = test_vae(net, test_loader, recon_weight, kl_weight, nn_type)

        writer.add_scalar('LogLoss/train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_train', np.log(train_recon_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_train', np.log(train_kld_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/test', np.log(test_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_test', np.log(test_recon_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_test', np.log(test_kld_loss / len(test_loader.dataset)), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + desc + '_' + now + '.pt')


def test_vae(net, test_loader, recon_weight, kl_weight, nn_type):
    net.eval()
    net = net.to(device)
    test_loss = 0.
    recon_loss = 0.
    kld_loss = 0.
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = net(data)

            # sum up batch loss
            batch_test_loss, batch_recon_loss, batch_kld_loss = loss_function_vae(recon, data, mu, log_var,
                                                                                  recon_weight, kl_weight, nn_type)
            test_loss += batch_test_loss.item()
            recon_loss += batch_recon_loss.item()
            kld_loss += batch_kld_loss.item()

    print('Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))

    return test_loss, recon_loss, kld_loss


# return reconstruction error + KL divergence losses
def loss_function_vae(recon_x, x, mu, log_var, recon_weight, kl_weight, nn_type):
    if nn_type == 'conv':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') * recon_weight
    else:
        recon_loss = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum') * recon_weight
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * kl_weight
    return recon_loss + KLD, recon_loss, KLD


def train_total_vae(net, train_loader, test_loader, epochs, optimizer, recon_weight=1., kl_weight=1., dataset='MNIST',
                    nn_type='conv', desc=''):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format(dataset + '_VAE_' + desc + '_' + now))
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_recon_loss = 0.
        train_kld_loss = 0.
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = net(data)
            batch_loss, batch_recon_loss, batch_kld_loss = loss_function_vae(recon_batch, targets, mu, log_var,
                                                                             recon_weight,
                                                                             kl_weight, nn_type)

            batch_loss.backward()
            train_loss += batch_loss.item()
            train_recon_loss += batch_recon_loss.item()
            train_kld_loss += batch_kld_loss.item()

            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

        test_loss, test_recon_loss, test_kld_loss = test_vae(net, test_loader, recon_weight, kl_weight, nn_type)

        writer.add_scalar('LogLoss/train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_train', np.log(train_recon_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_train', np.log(train_kld_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/test', np.log(test_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_test', np.log(test_recon_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_test', np.log(test_kld_loss / len(test_loader.dataset)), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + desc + '_' + now + '.pt')


def test_total_vae(net, test_loader, recon_weight, kl_weight, nn_type):
    net.eval()
    net = net.to(device)
    test_loss = 0.
    recon_loss = 0.
    kld_loss = 0.
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            recon, mu, log_var = net(data)

            # sum up batch loss
            batch_test_loss, batch_recon_loss, batch_kld_loss = loss_function_vae(recon, targets, mu, log_var,
                                                                                  recon_weight, kl_weight, nn_type)
            test_loss += batch_test_loss.item()
            recon_loss += batch_recon_loss.item()
            kld_loss += batch_kld_loss.item()

    print('Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))

    return test_loss, recon_loss, kld_loss




'''from constants import *
from datetime import datetime
from torch.nn import MSELoss
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train_vae(net, train_loader, test_loader, epochs, optimizer, recon_weight=1., kl_weight=1., dataset='MNIST'):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format(dataset + '_VAE_' + now))
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        recon_loss = 0.
        kld_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = net(data)
            batch_loss, batch_recon_loss, batch_kld_loss = loss_function_vae(recon_batch, data, mu, log_var,
                                                                             recon_weight,
                                                                             kl_weight)

            batch_loss.backward()

            #  Gradients norm plotting
            for name, param in net.named_parameters():
                writer.add_scalar(name, np.linalg.norm(param.grad.data.cpu()),
                                  (batch_idx + 1) + (batch_idx + 1) * (epoch + 1))

            train_loss += batch_loss.item()
            recon_loss += batch_recon_loss.item()
            kld_loss += batch_kld_loss.item()

            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch + 1, train_loss / len(train_loader.dataset)))

        test_loss = test_vae(net, test_loader, recon_weight, kl_weight)

        writer.add_scalar('Loss/log_train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('Loss/log_recon_train', np.log(recon_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('Loss/log_kld_train', np.log(kld_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('Loss/log_test', np.log(test_loss / len(test_loader.dataset)), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + now + '.pt')


def test_vae(net, test_loader, recon_weight, kl_weight):
    net.eval()
    net = net.to(device)
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon, mu, log_var = net(data)

            # sum up batch loss
            test_loss += loss_function_vae(recon, data, mu, log_var, recon_weight, kl_weight)[0].item()

    print('Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))

    return test_loss


# return reconstruction error + KL divergence losses
def loss_function_vae(recon_x, x, mu, log_var, recon_weight, kl_weight):
    recon = F.mse_loss(recon_x, x, reduction='sum') * recon_weight
    # recon = F.binary_cross_entropy(recon_x, x, reduction='sum') * recon_weight
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * kl_weight
    return recon + KLD, recon, KLD


def train_ae(net, train_loader, test_loader, epochs, optimizer, dataset):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format('AE_' + dataset + '_' + now))
    net = net.to(device)
    net.train()
    mse_loss = MSELoss()
    for epoch in range(epochs):
        train_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            output = net(data)
            loss = mse_loss(output, data)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))
        test_loss = test_ae(net, test_loader)

        #  Backup save
        if epoch % 150 == 0 and epoch != 0:
            # Save the model at current date and time
            torch.save(net.state_dict(), MODELS_ROOT + 'AE_' + dataset + '_' + now + '_backup.pt')

        writer.add_scalar('Loss/log_train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('Loss/log_test', np.log(test_loss / len(test_loader.dataset)), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + 'AE_' + dataset + '_' + now + '.pt')


def test_ae(net, test_loader):
    net.eval()
    net = net.to(device)
    test_loss = 0
    mse_loss = MSELoss()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = net(data)

            test_loss += mse_loss(output, data).item()

    print('Test set loss: {:.8f}'.format(test_loss))
    return test_loss


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

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))
        test_loss = test_mapper(net, test_loader)

        #  Backup save
        if epoch % 50 == 0 and epoch != 0:
            # Save the model at current date and time
            torch.save(net.state_dict(), MODELS_ROOT + 'Mapper_' + now + '_backup.pt')

        writer.add_scalar('Loss/log_train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('Loss/log_test', np.log(test_loss / len(test_loader.dataset)), epoch)

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

    print('Test set loss: {:.8f}'.format(test_loss))
    return test_loss'''
