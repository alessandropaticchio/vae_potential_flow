from constants import *
from models import Mapper, ConvMapper
from training import train_mapper
import torch.optim as optim

# python -m tensorboard.main --logdir=runs

mapper_type = 'lin'
subset = True

if subset:
    train_dataset = torch.load(DATA_ROOT + '/subsets/encoded_mapped/' + 'training.pt')
    test_dataset = torch.load(DATA_ROOT + '/subsets/encoded_mapped/' + 'test.pt')
else:
    train_dataset = torch.load(DATA_ROOT + '/encoded_mapped/' + 'training.pt')
    test_dataset = torch.load(DATA_ROOT + '/encoded_mapped/' + 'test.pt')

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if mapper_type == 'conv':
    mapper = ConvMapper(mnist_encoded_size=14, fashion_mnist_encoded_size=14)
else:
    mapper = Mapper(h_sizes=[1568, 1568, 1568, 1568])

lr = 1e-3
optimizer = optim.Adam(mapper.parameters(), lr=lr)

mapper.load_state_dict(torch.load('/Users/alessandro.paticchio/Desktop/vae_potential_flow/models/Mapper_2021-01-17 09:55:18.164844.pt'))

train_mapper(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=300, optimizer=optimizer)
