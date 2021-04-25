from constants import *
from datetime import datetime
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def train(net, train_loader, test_loader, epochs, optimizer):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format('Mapper_' + now))
    net = net.to(device)
    net.train()
    mse_loss = MSELoss()
    for epoch in range(epochs):
        train_loss = 0.
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = net(data)
            loss = mse_loss(output, target)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))
        test_loss = test(net, test_loader)

        writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/test', test_loss / len(train_loader.dataset), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + 'Mapper_' + now + '.pt')


def test(net, test_loader):
    net.eval()
    net = net.to(device)
    test_loss = 0
    mse_loss = MSELoss()
    with torch.no_grad():
        for data, target, _ in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)

            test_loss += mse_loss(output, target).item()

    print('====> Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))
    return test_loss


def train_vae(net, train_loader, test_loader, epochs, optimizer, recon_weight=1., kl_weight=1., dataset='MNIST',
              nn_type='conv', is_L1=False, power=0, desc='', reg_weight=0):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format(dataset + '_VAE_' + str(desc) + '_' + now))
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_recon_loss = 0.
        train_kld_loss = 0.
        train_reg_loss = 0.
        for batch_idx, (data, strength) in enumerate(train_loader):
            data = data.to(device)
            strength = strength.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = net(data, strength)

            batch_loss, batch_recon_loss, batch_kld_loss, batch_reg_loss = loss_function_vae(recon_x=recon_batch,
                                                                                             x=data, strength=strength,
                                                                                             mu=mu, log_var=log_var,
                                                                                             recon_weight=recon_weight,
                                                                                             kl_weight=kl_weight,
                                                                                             reg_weight=reg_weight,
                                                                                             power=power,
                                                                                             nn_type=nn_type)
            # Adding code for L1 Regularisation
            if is_L1:

                l1_crit = nn.L1Loss(size_average=False)
                reg_loss = 0
                for param in net.parameters():
                    reg_loss += l1_crit(param, target=torch.zeros_like(param))

                factor = 0.00005
                batch_loss += factor * reg_loss

            batch_loss.backward()
            train_loss += batch_loss.item()
            train_recon_loss += batch_recon_loss.item()
            train_kld_loss += batch_kld_loss.item()
            train_reg_loss += batch_reg_loss.item()

            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

        test_loss, test_recon_loss, test_kld_loss, test_reg_loss = test_vae(net, test_loader, recon_weight, kl_weight,
                                                                            nn_type, reg_weight, power=power)

        writer.add_scalar('LogLoss/train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_train', np.log(train_recon_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_train', np.log(train_kld_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/reg_latent_train', np.log(train_reg_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/validation', np.log(test_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_validation', np.log(test_recon_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_validation', np.log(test_kld_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/reg_latent_validation', np.log(test_reg_loss / len(train_loader.dataset)), epoch)

        # backup save
        if epoch % 50 == 0 and epoch != 0:
            torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + str(desc) + '_' + now + '.pt')

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + str(desc) + '_' + now + '.pt')


def test_vae(net, test_loader, recon_weight, kl_weight, nn_type, reg_weight, power=0):
    net.eval()
    net = net.to(device)
    test_loss = 0.
    recon_loss = 0.
    kld_loss = 0.
    reg_loss = 0.
    with torch.no_grad():
        for (data, strength) in test_loader:
            data = data.to(device)
            strength = strength.to(device)

            recon, mu, log_var = net(data, strength)

            # sum up batch loss
            batch_test_loss, batch_recon_loss, batch_kld_loss, batch_reg_loss = loss_function_vae(recon_x=recon, x=data,
                                                                                                  strength=strength,
                                                                                                  mu=mu,
                                                                                                  log_var=log_var,
                                                                                                  recon_weight=recon_weight,
                                                                                                  kl_weight=kl_weight,
                                                                                                  nn_type=nn_type,
                                                                                                  reg_weight=reg_weight,
                                                                                                  power=power)
            test_loss += batch_test_loss.item()
            recon_loss += batch_recon_loss.item()
            kld_loss += batch_kld_loss.item()
            reg_loss += batch_reg_loss.item()

    print('Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))

    return test_loss, recon_loss, kld_loss, reg_loss


# return reconstruction error + KL divergence losses
def loss_function_vae(recon_x, x, strength, mu, log_var, recon_weight, kl_weight, nn_type, reg_weight=0, power=0):
    if nn_type == 'conv':
        if power > 1:
            recon_x = torch.pow(recon_x, power)
            x = torch.pow(x, power)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') * recon_weight
    else:
        recon_loss = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum') * recon_weight
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * kl_weight
    reg_latent_space = reg_weight * F.mse_loss(strength, torch.norm(mu, dim=1).unsqueeze(1), reduction='sum')
    return recon_loss + KLD + reg_latent_space, recon_loss, KLD, reg_latent_space


def train_unet_vae(net, train_loader, test_loader, epochs, optimizer, recon_weight=1., kl_weight=1., reg_weight=0,
                   dataset='MNIST',
                   power=0, nn_type='conv', desc=''):
    now = str(datetime.now())
    writer = SummaryWriter('runs/{}'.format(dataset + '_VAE_' + desc + '_' + now))
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_recon_loss = 0.
        train_kld_loss = 0.
        train_reg_loss = 0.
        for batch_idx, (data, targets, strengths) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            strengths = strengths.to(device)

            optimizer.zero_grad()

            recon_batch, mu, log_var = net(data)

            batch_loss, batch_recon_loss, batch_kld_loss, batch_reg_loss = loss_function_vae(recon_x=recon_batch,
                                                                                             x=targets,
                                                                                             strength=strengths,
                                                                                             mu=mu, log_var=log_var,
                                                                                             recon_weight=recon_weight,
                                                                                             kl_weight=kl_weight,
                                                                                             nn_type=nn_type,
                                                                                             reg_weight=reg_weight,
                                                                                             power=power)

            batch_loss.backward()
            train_loss += batch_loss.item()
            train_recon_loss += batch_recon_loss.item()
            train_kld_loss += batch_kld_loss.item()
            train_reg_loss += batch_reg_loss.item()

            optimizer.step()

        print('Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

        test_loss, test_recon_loss, test_kld_loss = test_unet_vae(net=net, test_loader=test_loader,
                                                                  recon_weight=recon_weight, kl_weight=kl_weight,
                                                                  nn_type=nn_type, power=power)

        writer.add_scalar('LogLoss/train', np.log(train_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_train', np.log(train_recon_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_train', np.log(train_kld_loss / len(train_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/validation', np.log(test_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/recon_validation', np.log(test_recon_loss / len(test_loader.dataset)), epoch)
        writer.add_scalar('LogLoss/kld_validation', np.log(test_kld_loss / len(test_loader.dataset)), epoch)

    # Save the model at current date and time
    torch.save(net.state_dict(), MODELS_ROOT + dataset + '_VAE_' + desc + '_' + now + '.pt')


def test_unet_vae(net, test_loader, recon_weight, kl_weight, nn_type, reg_weight=0, power=0):
    net.eval()
    net = net.to(device)
    test_loss = 0.
    recon_loss = 0.
    kld_loss = 0.
    reg_loss = 0.
    with torch.no_grad():
        for data, targets, strengths in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            strengths = strengths.to(device)

            recon_batch, mu, log_var = net(data)

            batch_test_loss, batch_recon_loss, batch_kld_loss, batch_reg_loss = loss_function_vae(recon_x=recon_batch,
                                                                                                  x=targets,
                                                                                                  strength=strengths,
                                                                                                  mu=mu,
                                                                                                  log_var=log_var,
                                                                                                  recon_weight=recon_weight,
                                                                                                  kl_weight=kl_weight,
                                                                                                  nn_type=nn_type,
                                                                                                  reg_weight=reg_weight,
                                                                                                  power=power)
            test_loss += batch_test_loss.item()
            recon_loss += batch_recon_loss.item()
            kld_loss += batch_kld_loss.item()
            reg_loss += batch_reg_loss.item()

    print('Test set loss: {:.8f}'.format(test_loss / len(test_loader.dataset)))

    return test_loss, recon_loss, kld_loss
