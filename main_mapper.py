from constants import *
from models import Mapper, ConvMapper
from training import train_mapper, test_mapper
import torch.optim as optim

# python -m tensorboard.main --logdir=runs

mapper_type = 'conv'
sampler = True

data_path = DATA_ROOT + '/mnist_encoded_mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

sampler_train = RandomSampler(data_source=train_dataset, replacement=True, num_samples=1000)
sampler_test = RandomSampler(data_source=test_dataset, replacement=True, num_samples=170)

batch_size = 100
if sampler:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=sampler_test)
else:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if mapper_type == 'conv':
    mapper = ConvMapper(potential_encoded_size=14, rays_encoded_size=14)
else:
    mapper = Mapper(h_sizes=[32, 32, 32])

lr = 1e-2
optimizer = optim.SGD(mapper.parameters(), lr=lr)

train_mapper(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer)