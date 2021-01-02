import torch

DATA_ROOT = './data/'
MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POTENTIAL_IMAGE_SIZE = 20
#  For DeConvVAE HIDDEN_SIZE = 576, for ConvVAE HIDDEN_SIZE = 3200
POTENTIAL_HIDDEN_SIZE = 3200
POTENTIAL_IMAGE_CHANNELS = 3
POTENTIAL_ROOT = 'potential_pic_data_1224_20/'
POTENTIAL_ENCODED_SIZE = 800

RAYS_IMAGE_SIZE = 50
#  For DeConvVAE HIDDEN_SIZE = 576, for ConvVAE HIDDEN_SIZE = 3200
RAYS_HIDDEN_SIZE = 3200
RAYS_IMAGE_CHANNELS = 3
RAYS_ROOT = 'rays_pic_data_1224_20/'
RAYS_ENCODED_SIZE = 5000

# python -m tensorboard.main --logdir=runs
