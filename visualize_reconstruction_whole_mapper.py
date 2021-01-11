from models import ConvWholeMapper
from constants import *
from utils import MyDataset
import matplotlib.pyplot as plt
import random
import torch

batch_size = 1

model_name = 'Mapper_2021-01-08 11:00:57.714363.pt'
model_path = MODELS_ROOT + model_name

whole_mapper = ConvWholeMapper(potential_encoded_size=POTENTIAL_ENCODED_IMAGE_SIZE[1],
                               rays_encoded_size=RAYS_ENCODED_IMAGE_SIZE[1])
whole_mapper.load_state_dict(torch.load(model_path))
whole_mapper.eval()

data_path = DATA_ROOT + '/real_data/'

potential_train_dataset = torch.load(data_path + POTENTIAL_ROOT + 'training_potential.pt')
potential_test_dataset = torch.load(data_path + POTENTIAL_ROOT + 'test_potential.pt')

rays_train_dataset = torch.load(data_path + RAYS_ROOT + 'training_rays.pt')
rays_test_dataset = torch.load(data_path + RAYS_ROOT + 'test_rays.pt')

whole_train_dataset = MyDataset(x=potential_train_dataset, y=rays_train_dataset)
whole_test_dataset = MyDataset(x=potential_test_dataset, y=rays_test_dataset)

rand_sample_idx = random.randint(0, 500)
rand_sample = whole_train_dataset.X[rand_sample_idx].unsqueeze(0)

rand_sample_prime = whole_mapper(rand_sample)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(rand_sample.squeeze().permute(1, 2, 0))

plt.subplot(1, 3, 2)
plt.title('Reconstruction')
plt.imshow(rand_sample_prime.squeeze().detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Target')
plt.imshow(whole_train_dataset.y[rand_sample_idx].detach().permute(1, 2, 0).numpy(), cmap='gray')

plt.show()
