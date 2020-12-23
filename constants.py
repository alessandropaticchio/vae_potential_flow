import torch

DATA_ROOT = './data/'
MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POTENTIAL_IMAGE_SIZE = 20
POTENTIAL_HIDDEN_SIZE = 3200
POTENTIAL_IMAGE_CHANNELS = 3

RAYS_IMAGE_SIZE = 20
RAYS_HIDDEN_SIZE = 3200
RAYS_IMAGE_CHANNELS = 3

# python -m tensorboard.main --logdir=runs