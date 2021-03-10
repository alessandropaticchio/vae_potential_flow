import torch

colab = False

if colab:
    DATA_ROOT = '/content/gdrive/MyDrive/Colab_Notebooks/data/'
else:
    DATA_ROOT = './data/'

MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_SIZE = 16 * 149 * 149
LATENT_SIZE = 20

POTENTIAL_IMAGE_SIZE = 300
#  For DeConvVAE HIDDEN_SIZE = 576, for ConvVAE HIDDEN_SIZE = 3200
# POTENTIAL_HIDDEN_SIZE = 3200
POTENTIAL_IMAGE_CHANNELS = 3
# POTENTIAL_ROOT = 'potential_pic_data_1224_20/'
# POTENTIAL_ENCODED_IMAGE_SIZE = [8, 10, 10]
# POTENTIAL_ENCODED_SIZE = POTENTIAL_ENCODED_IMAGE_SIZE[0] * POTENTIAL_ENCODED_IMAGE_SIZE[1] * \
#                         POTENTIAL_ENCODED_IMAGE_SIZE[2]

RAYS_IMAGE_SIZE = 300
#  For DeConvVAE HIDDEN_SIZE = 576, for ConvVAE HIDDEN_SIZE = 3200
# RAYS_HIDDEN_SIZE = 36864
RAYS_IMAGE_CHANNELS = 3
# RAYS_ROOT = 'rays_pic_data_1224_20/'
# RAYS_ENCODED_IMAGE_SIZE = [8, 50, 50]
# RAYS_ENCODED_SIZE = RAYS_ENCODED_IMAGE_SIZE[0] * RAYS_ENCODED_IMAGE_SIZE[1] * RAYS_ENCODED_IMAGE_SIZE[2]

# python -m tensorboard.main --logdir=runs
