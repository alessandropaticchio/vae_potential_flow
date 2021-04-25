from constants import *
from models import Mapper
from training import train
import torch.optim as optim

data_path = DATA_ROOT + 'num=999_unscaled/mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

h0 = POTENTIAL_LATENT_SIZE * 2
h1 = RAYS_LATENT_SIZE * 2
h2 = RAYS_LATENT_SIZE * 2
h3 = RAYS_LATENT_SIZE * 2

mapper = Mapper(h_sizes=[h0, h1, h2, h3])

lr = 1e-3
optimizer = optim.Adam(mapper.parameters(), lr=lr)

train(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=1000, optimizer=optimizer)
