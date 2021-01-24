from constants import *
from models import DenseVAE, ConvVAE
from tqdm import tqdm
from utils import MyDataset
import torch


mapper_type = 'conv'
subset = True

if subset:
    mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/MNIST/training.pt')
    mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/MNIST/test.pt')
    fashion_mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/Fashion_MNIST/training.pt')
    fashion_mnist_test_dataset = torch.load(DATA_ROOT + 'subsets/Fashion_MNIST/test.pt')
else:
    from mnist_downloader import train_dataset as mnist_train_dataset, test_dataset as mnist_test_dataset
    from fashion_mnist_downloader import train_dataset as fashion_mnist_train_dataset, \
        test_dataset as fashion_mnist_test_dataset

mnist_model_name = 'MNIST_VAE_3layers_2021-01-22 12:13:27.636602.pt'
fashion_mnist_model_name = 'Fashion_MNIST_VAE_3layers_2021-01-22 12:15:48.234549.pt'
mnist_model_path = MODELS_ROOT + mnist_model_name
fashion_mnist_model_path = MODELS_ROOT + fashion_mnist_model_name

if mapper_type == 'conv':
    mnist_vae = ConvVAE(image_dim=28, hidden_size=HIDDEN_SIZE, latent_size=int(HIDDEN_SIZE / 2), image_channels=1)
else:
    mnist_vae = DenseVAE()

mnist_vae.load_state_dict(torch.load(mnist_model_path))
mnist_vae.eval()

if mapper_type == 'conv':
    fashion_mnist_vae = ConvVAE(image_dim=28, hidden_size=HIDDEN_SIZE, latent_size=int(HIDDEN_SIZE / 2),
                                image_channels=1)
else:
    fashion_mnist_vae = DenseVAE()

fashion_mnist_vae.load_state_dict(torch.load(fashion_mnist_model_path))
fashion_mnist_vae.eval()

mnist_train_targets = mnist_train_dataset.targets
mnist_test_targets = mnist_test_dataset.targets
fashion_mnist_train_targets = fashion_mnist_train_dataset.targets
fashion_mnist_test_targets = fashion_mnist_test_dataset.targets

targets = range(0, 10)

if mapper_type == 'conv':
    encoded_train_set_X = torch.empty((1, HIDDEN_SIZE))
    encoded_train_set_y = torch.empty((1, HIDDEN_SIZE))

    encoded_test_set_X = torch.empty((1, HIDDEN_SIZE))
    encoded_test_set_y = torch.empty((1, HIDDEN_SIZE))
else:
    encoded_train_set_X = torch.empty((1, 1, 32))
    encoded_train_set_y = torch.empty((1, 1, 32))

    encoded_test_set_X = torch.empty((1, 1, 32))
    encoded_test_set_y = torch.empty((1, 1, 32))

for target in targets:
    if subset:
        mnist_train_indeces = (mnist_train_targets.squeeze() == target).nonzero().flatten().tolist()
        mnist_test_indeces = (mnist_test_targets.squeeze() == target).nonzero().flatten().tolist()
        fashion_mnist_train_indeces = (fashion_mnist_train_targets.squeeze() == target).nonzero().flatten().tolist()
        fashion_mnist_test_indeces = (fashion_mnist_test_targets.squeeze() == target).nonzero().flatten().tolist()
    else:
        mnist_train_indeces = (mnist_train_targets == target).nonzero().flatten().tolist()
        mnist_test_indeces = (mnist_test_targets == target).nonzero().flatten().tolist()
        fashion_mnist_train_indeces = (fashion_mnist_train_targets == target).nonzero().flatten().tolist()
        fashion_mnist_test_indeces = (fashion_mnist_test_targets == target).nonzero().flatten().tolist()

    # Training set generation
    for i, idx in tqdm(enumerate(mnist_train_indeces), desc='Train set...'):
        if subset:
            mnist_sample = mnist_train_dataset.data[idx].unsqueeze(0).float()
        else:
            mnist_sample = mnist_train_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample -= torch.min(mnist_sample)
        mnist_sample /= torch.max(mnist_sample)
        if mapper_type == 'conv':
            mnist_sample_encoded_mean, mnist_sample_encoded_log_var = mnist_vae.encode(mnist_sample)
            mnist_sample_encoded = torch.cat((mnist_sample_encoded_mean, mnist_sample_encoded_log_var), 1)
        else:
            mnist_sample_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)

        try:
            if subset:
                fashion_mnist_sample = fashion_mnist_train_dataset.data[fashion_mnist_train_indeces[i]].unsqueeze(
                    0).float()
            else:
                fashion_mnist_sample = fashion_mnist_train_dataset.data[fashion_mnist_train_indeces[i]].unsqueeze(
                    0).unsqueeze(
                    0).float()
            fashion_mnist_sample -= torch.min(fashion_mnist_sample)
            fashion_mnist_sample /= torch.max(fashion_mnist_sample)
            if mapper_type == 'conv':
                fashion_mnist_sample_encoded_mean, fashion_mnist_sample_encoded_log_var = fashion_mnist_vae.encode(mnist_sample)
                fashion_mnist_sample_encoded = torch.cat((fashion_mnist_sample_encoded_mean, fashion_mnist_sample_encoded_log_var), 1)
            else:
                fashion_mnist_sample_encoded = fashion_mnist_vae.encode(fashion_mnist_sample).reshape(1, 1, 32)

            encoded_train_set_X = torch.cat((encoded_train_set_X, mnist_sample_encoded), 0)
            encoded_train_set_y = torch.cat((encoded_train_set_y, fashion_mnist_sample_encoded), 0)

        except IndexError:
            continue

    # Test set generation
    for i, idx in tqdm(enumerate(mnist_test_indeces), desc="Test set..."):
        if subset:
            mnist_sample = mnist_test_dataset.data[idx].unsqueeze(0).float()
        else:
            mnist_sample = mnist_test_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample -= torch.min(mnist_sample)
        mnist_sample /= torch.max(mnist_sample)
        if mapper_type == 'conv':
            mnist_sample_encoded_mean, mnist_sample_encoded_log_var = mnist_vae.encode(mnist_sample)
            mnist_sample_encoded = torch.cat((mnist_sample_encoded_mean, mnist_sample_encoded_log_var), 1)
        else:
            mnist_sample_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)

        try:
            if subset:
                fashion_mnist_sample = fashion_mnist_test_dataset.data[fashion_mnist_test_indeces[i]].unsqueeze(
                    0).float()
            else:
                fashion_mnist_sample = fashion_mnist_test_dataset.data[fashion_mnist_test_indeces[i]].unsqueeze(
                    0).unsqueeze(0).float()
            fashion_mnist_sample -= torch.min(fashion_mnist_sample)
            fashion_mnist_sample /= torch.max(fashion_mnist_sample)
            if mapper_type == 'conv':
                fashion_mnist_sample_encoded_mean, fashion_mnist_sample_encoded_log_var = fashion_mnist_vae.encode(mnist_sample)
                fashion_mnist_sample_encoded = torch.cat((fashion_mnist_sample_encoded_mean, fashion_mnist_sample_encoded_log_var), 1)
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

encoded_test_set = MyDataset(x=encoded_test_set_X, y=encoded_test_set_y)
encoded_train_set = MyDataset(x=encoded_train_set_X, y=encoded_train_set_y)

if subset:
    torch.save(encoded_train_set, DATA_ROOT + 'subsets/encoded_mapped/training.pt')
    torch.save(encoded_test_set, DATA_ROOT + 'subsets/encoded_mapped/test.pt')
else:
    torch.save(encoded_train_set, DATA_ROOT + '/mnist_encoded_mapped/training.pt')
    torch.save(encoded_test_set, DATA_ROOT + '/mnist_encoded_mapped/test.pt')
