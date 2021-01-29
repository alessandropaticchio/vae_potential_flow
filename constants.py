import torch

DATA_ROOT = './data/'
MODELS_ROOT = './models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_SIZE = 12 * 12 * 16
LATENT_SIZE = 20