from constants import *
from models import Mapper
from training import train_mapper, test_mapper
import torch.optim as optim

# python -m tensorboard.main --logdir=runs

data_path = DATA_ROOT + '/encoded_mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mapper = Mapper(h_sizes=[32, 32, 32])
lr = 1e-2
optimizer = optim.SGD(mapper.parameters(), lr=lr)

train_mapper(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer)