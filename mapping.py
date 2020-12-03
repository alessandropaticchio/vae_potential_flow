from constants import *
from models import LinearVAE
from tqdm import tqdm
from mnist_downloader import train_dataset as mnist_train_dataset, test_dataset as mnist_test_dataset
from fashion_mnist_downloader import train_dataset as fashion_mnist_train_dataset, \
    test_dataset as fashion_mnist_test_dataset
from utils import EncodedDataset
import torch


mnist_model_name = 'MNIST_VAE_2020-12-02 11:05:21.291726.pt'
fashion_mnist_model_name = 'Fashion_MNIST_VAE_2020-12-02 12:02:42.435081.pt'
mnist_model_path = MODELS_ROOT + mnist_model_name
fashion_mnist_model_path = MODELS_ROOT + fashion_mnist_model_name

mnist_vae = LinearVAE()
mnist_vae.load_state_dict(torch.load(mnist_model_path))
mnist_vae.eval()

fashion_mnist_vae = LinearVAE()
fashion_mnist_vae.load_state_dict(torch.load(fashion_mnist_model_path))
fashion_mnist_vae.eval()

mnist_train_targets = mnist_train_dataset.targets
mnist_test_targets = mnist_test_dataset.targets
fashion_mnist_train_targets = fashion_mnist_train_dataset.targets
fashion_mnist_test_targets = fashion_mnist_test_dataset.targets

targets = range(0, 9)

encoded_train_set_X = torch.empty((1, 2, 16))
encoded_train_set_y = torch.empty((1, 2, 16))

encoded_test_set_X = torch.empty((1, 2, 16))
encoded_test_set_y = torch.empty((1, 2, 16))

for target in targets:
    mnist_train_indeces = (mnist_train_targets == target).nonzero().flatten().tolist()
    mnist_test_indeces = (mnist_test_targets == target).nonzero().flatten().tolist()
    fashion_mnist_train_indeces = (fashion_mnist_train_targets == target).nonzero().flatten().tolist()
    fashion_mnist_test_indeces = (fashion_mnist_test_targets == target).nonzero().flatten().tolist()

    # Training set generation
    for i, idx in tqdm(enumerate(mnist_train_indeces), desc='Train set...'):
        mnist_sample = mnist_train_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample_encoded = mnist_vae.encode(mnist_sample)

        try:
            fashion_mnist_sample = fashion_mnist_train_dataset.data[fashion_mnist_train_indeces[i]].unsqueeze(0).unsqueeze(0).float()
        except IndexError:
            continue
        fashion_mnist_sample_encoded = fashion_mnist_vae.encode(mnist_sample)

        encoded_train_set_X = torch.cat((encoded_train_set_X, mnist_sample_encoded), 0)
        encoded_train_set_y = torch.cat((encoded_train_set_y, fashion_mnist_sample_encoded), 0)


    # Test set generation
    for i, idx in tqdm(enumerate(mnist_test_indeces), desc="Test set..."):
        mnist_sample = mnist_test_dataset.data[idx].unsqueeze(0).unsqueeze(0).float()
        mnist_sample_encoded = mnist_vae.encode(mnist_sample)

        try:
            fashion_mnist_sample = fashion_mnist_test_dataset.data[fashion_mnist_test_indeces[i]].unsqueeze(0).unsqueeze(0).float()
        except:
            continue
        fashion_mnist_sample_encoded = fashion_mnist_vae.encode(mnist_sample)

        encoded_test_set_X = torch.cat((encoded_test_set_X, mnist_sample_encoded), 0)
        encoded_test_set_y = torch.cat((encoded_test_set_y, fashion_mnist_sample_encoded), 0)

#  First tensor is meaningless
encoded_train_set_X = encoded_train_set_X[1:]
encoded_train_set_y = encoded_train_set_y[1:]
encoded_test_set_X = encoded_test_set_X[1:]
encoded_test_set_y = encoded_test_set_y[1:]

encoded_test_set = EncodedDataset(x=encoded_test_set_X, y=encoded_test_set_y)
encoded_train_set = EncodedDataset(x=encoded_train_set_X, y=encoded_train_set_y)

torch.save(encoded_train_set, DATA_ROOT + '/encoded_mapped/training.pt')
torch.save(encoded_test_set, DATA_ROOT + '/encoded_mapped/test.pt')
