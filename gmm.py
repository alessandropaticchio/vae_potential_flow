import torch
import torch.nn.functional as F
import numpy as np


# loss
def kld_loss(outputs, targets, freebits=1):
    prior_mu = targets
    q_mu = outputs['mu']
    q_logvar = outputs['log_var']
    kld = -1 - q_logvar + (q_mu - prior_mu) ** 2 + q_logvar.exp()
    # kld_free_bits = F.softplus(kld_matrix - 2 * freebits) + 2 * freebits
    # kld = 0.5 * kld_free_bits.mean(0).sum()
    kld_real = 0.5 * kld.mean(0).sum()
    return kld, kld_real


# set batch
def set_batch():
    std = 0.1
    batch_size = 10
    targets = torch.FloatTensor(np.arange(batch_size) / batch_size)
    outputs = dict(
        mu=targets + np.random.normal(scale=std, size=batch_size),
        log_var=torch.log(torch.full((batch_size,), std ** 2))
    )
    return outputs, targets


# test
outputs, targets = set_batch()
print(kld_loss(outputs, targets)[1])