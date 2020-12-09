from models import LinearVAE, Mapper
from constants import *
from mnist_downloader import train_dataset as mnist_train_dataset
import random
import matplotlib.pyplot as plt


mnist_model_name = 'MNIST_VAE_2020-12-07 09:50:15.947882.pt'
fashion_mnist_model_name = 'Fashion_MNIST_VAE_2020-12-07 10:02:43.350691.pt'
mapper_model_name = 'Mapper_2020-12-07 11:28:14.737913.pt'
mnist_model_path = MODELS_ROOT + mnist_model_name
fashion_mnist_model_path = MODELS_ROOT + fashion_mnist_model_name
mapper_model_path = MODELS_ROOT + mapper_model_name

mnist_vae = LinearVAE()
mnist_vae.load_state_dict(torch.load(mnist_model_path))
mnist_vae.eval()

fashion_mnist_vae = LinearVAE()
fashion_mnist_vae.load_state_dict(torch.load(fashion_mnist_model_path))
fashion_mnist_vae.eval()

mapper = Mapper(h_sizes=[32, 32, 32])
mapper.load_state_dict(torch.load(mapper_model_path))
mapper.eval()

for i in range(1, 20):
    rand_sample_idx = random.randint(10000, 12000)
    mnist_sample = mnist_train_dataset.data[rand_sample_idx].unsqueeze(0).unsqueeze(0).float()
    mnist_sample -= torch.min(mnist_sample)
    mnist_sample /= torch.max(mnist_sample)
    mnist_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)
    mapping = mapper(mnist_encoded).reshape(1, 2, 16)
    fashion_mnist_decoded = fashion_mnist_vae.decode(mapping)

    plt.figure()
    plt.imshow(mnist_sample[0].reshape(28, 28).detach().numpy())

    plt.figure()
    plt.imshow(fashion_mnist_decoded.reshape(28, 28).detach().numpy())


plt.show()

