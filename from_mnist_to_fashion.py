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


mnist_model_name = 'MNIST_VAE_3layers_2021-01-22 12:13:27.636602.pt'
fashion_mnist_model_name = 'Fashion_MNIST_VAE_3layers_2021-01-22 12:15:48.234549.pt'
mapper_model_name = 'Mapper_2021-01-22 12:19:46.681213.pt'
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
    mapper = Mapper(h_sizes=[HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE])
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
        mapping_mean, mapping_log_var = mapping[:, :int(HIDDEN_SIZE/2)], mapping[:, int(HIDDEN_SIZE/2):]
        fashion_mnist_decoded = fashion_mnist_vae.decode(mean=mapping_mean, log_var=mapping_log_var)
    else:
        mnist_encoded = mnist_vae.encode(mnist_sample).reshape(1, 1, 32)
        mapping = mapper(mnist_encoded).reshape(1, 2, 16)
        fashion_mnist_decoded = fashion_mnist_vae.decode(z=mapping)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('MNIST Original')
    plt.imshow(mnist_sample[0].reshape(28, 28).detach().numpy())

    plt.subplot(1, 2, 2)
    plt.title('Fashion MNIST Reconstruction')
    plt.imshow(fashion_mnist_decoded.reshape(28, 28).detach().numpy())


plt.show()

