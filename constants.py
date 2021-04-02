import torch

colab = False

if colab:
    DATA_ROOT = '/content/gdrive/MyDrive/data/'
else:
    DATA_ROOT = './data/'

MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POTENTIAL_IMAGE_SIZE = 100
POTENTIAL_IMAGE_CHANNELS = 3
POTENTIAL_HIDDEN_SIZE = 16 * 48 * 48
POTENTIAL_LATENT_SIZE = 20

RAYS_IMAGE_SIZE = 100
RAYS_IMAGE_CHANNELS = 3
RAYS_HIDDEN_SIZE = 16 * 48 * 48
RAYS_LATENT_SIZE = 20

# python -m tensorboard.main --logdir=runs
