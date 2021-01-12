from constants import *
from models import LinearVAE, ConvVAE
from tqdm import tqdm
from mnist_downloader import train_dataset as mnist_train_dataset, test_dataset as mnist_test_dataset
from fashion_mnist_downloader import train_dataset as fashion_mnist_train_dataset, \
    test_dataset as fashion_mnist_test_dataset
from utils import EncodedDataset
import torch

mapper_type = 'conv'

mnist_model_name = 'MNIST_VAE_2021-01-11 22:54:43.183412.pt'
fashion_mnist_model_name = 'Fashion_MNIST_VAE_2021-01-11 16:34:22.522792.pt'
mnist_model_path = MODELS_ROOT + mnist_model_name
fashion_mnist_model_path = MODELS_ROOT + fashion_mnist_model_name

if mapper_type == 'conv':
    hidden_size = 3136
    mnist_vae = ConvVAE(image_dim=28, hidden_size=hidden_size, latent_size=int(hidden_size / 2), image_channels=1)
else:
    mnist_vae = LinearVAE()

mnist_vae.load_state_dict(torch.load(mnist_model_path))
mnist_vae.eval()

if mapper_type == 'conv':
    hidden_size = 3136
    fashion_mnist_vae = ConvVAE(image_dim=28, hidden_size=hidden_size, latent_size=int(hidden_size / 2), image_channels=1)
else:
    fashion_mnist_vae = LinearVAE()

fashion_mnist_vae.load_state_dict(torch.load(fashion_mnist_model_path))
fashion_mnist_vae.eval()

mnist_train_targets = mnist_train_dataset.targets
mnist_test_targets = mnist_test_dataset.targets
fashion_mnist_train_targets = fashion_mnist_train_dataset.targets
fashion_mnist_test_targets = fashion_mnist_test_dataset.targets

targets = range(0, 9)

if mapper_type == 'conv':
    encoded_train_set_X = torch.empty((1, 16, 14, 14))
    encoded_train_set_y = torch.empty((1, 16, 14, 14))

    encoded_test_set_X = torch.empty((1, 16, 14, 14))
    encoded_test_set_y = torch.empty((1, 16, 14, 14))
else:
    encoded_train_set_X = torch.empty((1, 1, 32))
    encoded_train_set_y = torch.empty((1, 1, 32))

    encoded_test_set_X = torch.empty((1, 1, 32))
    encoded_test_set_y = torch.empty((1, 1, 32))

for target in targets:
    mnist_train_indeces = (mnist_train_targets == target).nonzero().flatten().tolist()
    mnist_test_indeces = (mnist_test_targets == target).nonzero().flatten().tolist()
    fashion_mnist_train_indeces = (fashion_mnist_train_targets == target).nonzero().flatten().tolist()
    fashion_mnist_test_indeces = (fashion_mnist_test_targets == target).nonzero().flatten().tolist()

    # Training set generation
    for i, idx in tqdm(enumerate(mnist_train_indeces), desc='Train set...'):
        mnist_sample = mnist_train_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample -= torch.min(mnist_sample)
        mnist_sample /= torch.max(mnist_sample)
        if mapper_type == 'conv':
            mnist_sample_encoded, _, _ = mnist_vae.encode(mnist_sample)
            mnist_sample_encoded = mnist_sample_encoded.reshape(1, 16, 14, 14)
        else:
            mnist_sample_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)

        try:
            fashion_mnist_sample = fashion_mnist_train_dataset.data[fashion_mnist_train_indeces[i]].unsqueeze(0).unsqueeze(0).float()
            fashion_mnist_sample -= torch.min(fashion_mnist_sample)
            fashion_mnist_sample /= torch.max(fashion_mnist_sample)
            if mapper_type == 'conv':
                fashion_mnist_sample_encoded, _, _ = fashion_mnist_vae.encode(fashion_mnist_sample)
                fashion_mnist_sample_encoded = fashion_mnist_sample_encoded.reshape(1, 16, 14, 14)
            else:
                fashion_mnist_sample_encoded = fashion_mnist_vae.encode(fashion_mnist_sample).reshape(1, 1, 32)

            encoded_train_set_X = torch.cat((encoded_train_set_X, mnist_sample_encoded), 0)
            encoded_train_set_y = torch.cat((encoded_train_set_y, fashion_mnist_sample_encoded), 0)

        except IndexError:
            continue

    # Test set generation
    for i, idx in tqdm(enumerate(mnist_test_indeces), desc="Test set..."):
        mnist_sample = mnist_test_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample -= torch.min(mnist_sample)
        mnist_sample /= torch.max(mnist_sample)
        if mapper_type == 'conv':
            mnist_sample_encoded, _, _ = mnist_vae.encode(mnist_sample)
            mnist_sample_encoded = mnist_sample_encoded.reshape(1, 16, 14, 14)
        else:
            mnist_sample_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)

        try:
            fashion_mnist_sample = fashion_mnist_test_dataset.data[fashion_mnist_test_indeces[i]].unsqueeze(0).unsqueeze(0).float()
            fashion_mnist_sample -= torch.min(fashion_mnist_sample)
            fashion_mnist_sample /= torch.max(fashion_mnist_sample)
            if mapper_type == 'conv':
                fashion_mnist_sample_encoded, _, _ = fashion_mnist_vae.encode(fashion_mnist_sample)
                fashion_mnist_sample_encoded = fashion_mnist_sample_encoded.reshape(1, 16, 14, 14)
            else:
                fashion_mnist_sample_encoded = fashion_mnist_vae.encode(fashion_mnist_sample).reshape(1, 1, 32)

            encoded_test_set_X = torch.cat((encoded_test_set_X, mnist_sample_encoded), 0)
            encoded_test_set_y = torch.cat((encoded_test_set_y, fashion_mnist_sample_encoded), 0)
        except IndexError:
            continue


#  First tensor is meaningless
encoded_train_set_X = encoded_train_set_X[1:]
encoded_train_set_y = encoded_train_set_y[1:]
encoded_test_set_X = encoded_test_set_X[1:]
encoded_test_set_y = encoded_test_set_y[1:]

encoded_test_set = EncodedDataset(x=encoded_test_set_X, y=encoded_test_set_y)
encoded_train_set = EncodedDataset(x=encoded_train_set_X, y=encoded_train_set_y)

torch.save(encoded_train_set, DATA_ROOT + '/mnist_encoded_mapped/training.pt')
torch.save(encoded_test_set, DATA_ROOT + '/mnist_encoded_mapped/test.pt')
