from models import LinearVAE, Mapper, ConvVAE, ConvMapper
from constants import *
import random
import matplotlib.pyplot as plt

subset = True
mapper_type = 'lin'
vae_type = 'conv'

if subset:
    mnist_train_dataset = torch.load(DATA_ROOT + 'subsets/MNIST/training.pt')
else:
    from mnist_downloader import train_dataset as mnist_train_dataset


mnist_model_name = 'MNIST_VAE_2021-01-16 15:43:45.546348.pt'
fashion_mnist_model_name = 'Fashion_MNIST_VAE_2021-01-16 15:59:52.537353.pt'
mapper_model_name = 'Mapper_2021-01-17 10:00:46.999054.pt'
mnist_model_path = MODELS_ROOT + mnist_model_name
fashion_mnist_model_path = MODELS_ROOT + fashion_mnist_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name

if vae_type == 'conv':
    mnist_vae = ConvVAE(image_dim=28, hidden_size=HIDDEN_SIZE, latent_size=int(HIDDEN_SIZE/2), image_channels=1)
else:
    mnist_vae = LinearVAE()
mnist_vae.load_state_dict(torch.load(mnist_model_path))
mnist_vae.eval()

if vae_type == 'conv':
    fashion_mnist_vae = ConvVAE(image_dim=28, hidden_size=HIDDEN_SIZE, latent_size=int(HIDDEN_SIZE/2), image_channels=1)
else:
    fashion_mnist_vae = LinearVAE()
fashion_mnist_vae.load_state_dict(torch.load(fashion_mnist_model_path))
fashion_mnist_vae.eval()

if mapper_type == 'conv':
    mapper = ConvMapper(mnist_encoded_size=14, fashion_mnist_encoded_size=14)
else:
    mapper = Mapper(h_sizes=[1568, 1568, 1568, 1568])
mapper.load_state_dict(torch.load(mapper_model_path))
mapper.eval()

for i in range(1, 20):
    rand_sample_idx = random.randint(0, 1000)
    if subset:
        mnist_sample = mnist_train_dataset.data[rand_sample_idx].unsqueeze(0).float()
    else:
        mnist_sample = mnist_train_dataset.data[rand_sample_idx].unsqueeze(0).unsqueeze(0).float()
    mnist_sample -= torch.min(mnist_sample)
    mnist_sample /= torch.max(mnist_sample)
    if vae_type == 'conv':
        mnist_encoded_mean, mnist_encoded_log_var = mnist_vae.encode(mnist_sample)
        mnist_encoded = torch.cat((mnist_encoded_mean, mnist_encoded_log_var), 1)
        mapping = mapper(mnist_encoded)
        mapping_mean, mapping_log_var = mapping[:, :784], mapping[:, 784:]
    else:
        mnist_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)
        mapping = mapper(mnist_encoded).reshape(1, 2, 16)
    fashion_mnist_decoded = fashion_mnist_vae.decode(mean=mapping_mean, log_var=mapping_log_var)

    plt.figure()
    plt.imshow(mnist_sample[0].reshape(28, 28).detach().numpy())

    plt.figure()
    plt.imshow(fashion_mnist_decoded.reshape(28, 28).detach().numpy())


plt.show()

