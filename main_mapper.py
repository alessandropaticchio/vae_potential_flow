from constants import *
from models import Mapper, ConvMapper
from training import train_mapper, test_mapper
import torch.optim as optim

# python -m tensorboard.main --logdir=runs

mapper_type = 'conv'

data_path = DATA_ROOT + '/mnist_encoded_mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if mapper_type == 'conv':
    mapper = ConvMapper(potential_encoded_size=14, rays_encoded_size=14)
else:
    mapper = Mapper(h_sizes=[32, 32, 32])

lr = 1e-2
optimizer = optim.SGD(mapper.parameters(), lr=lr)

train_mapper(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer)