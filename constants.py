import torch

DATA_ROOT = './data/'
MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POTENTIAL_IMAGE_SIZE = 100
POTENTIAL_HIDDEN_SIZE = 800
POTENTIAL_IMAGE_CHANNELS = 1

RAYS_IMAGE_SIZE = 100
RAYS_HIDDEN_SIZE = 800
RAYS_IMAGE_CHANNELS = 1

# python -m tensorboard.main --logdir=runs