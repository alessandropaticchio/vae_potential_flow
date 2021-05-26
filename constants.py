import torch

colab = False

if colab:
    DATA_ROOT = '/content/gdrive/MyDrive/data/'
else:
    DATA_ROOT = './data/'

MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STRENGTHS = [0.01, 0.1, 0.2, 0.05, 0.07, 0.03, 0.3]
# STRENGTHS = [0.01, 0.1, 0.3]

POTENTIAL_IMAGE_SIZE = 100
POTENTIAL_IMAGE_CHANNELS = 3
POTENTIAL_HIDDEN_SIZE = 16 * 48 * 48
POTENTIAL_LATENT_SIZE = 20

RAYS_IMAGE_SIZE = 100
RAYS_IMAGE_CHANNELS = 3
RAYS_HIDDEN_SIZE = 16 * 48 * 48
RAYS_LATENT_SIZE = 20


H0 = POTENTIAL_LATENT_SIZE * 2
H1 = RAYS_LATENT_SIZE * 2
H2 = RAYS_LATENT_SIZE * 2
H3 = RAYS_LATENT_SIZE * 2
H4 = RAYS_LATENT_SIZE * 2
H5 = RAYS_LATENT_SIZE * 2

MAPPER_LAYERS = [H0,  H1,  H2,  H3,  H4,  H5]


COLORS = ['#ff5050', '#ff9966', '#99ff66', '#00cc66', '#9966ff', '#3366ff', '#cc3300', '#003300', '#666699', '#003366']
COLORS = COLORS[:len(STRENGTHS)]

STRENGTHS_COLORS = dict(zip(STRENGTHS, COLORS))

# python -m tensorboard.main --logdir=runs
