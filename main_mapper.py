from constants import *
from models import Mapper
from training import train_mapper
import torch.optim as optim

data_path = DATA_ROOT + '/encoded_mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mapper = Mapper(h_sizes=[POTENTIAL_ENCODED_SIZE, POTENTIAL_ENCODED_SIZE, RAYS_ENCODED_SIZE, RAYS_ENCODED_SIZE])
lr = 1e-3
optimizer = optim.Adam(mapper.parameters(), lr=lr)

train_mapper(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=50, optimizer=optimizer)