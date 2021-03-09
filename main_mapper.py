from constants import *
from models import Mapper
from training import train
import torch.optim as optim

data_path = DATA_ROOT + '/mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

# Fetch first sample
# train_dataset.y = train_dataset.y[0, :]
# train_dataset.X = train_dataset.X[0, :]

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mapper = Mapper(h_sizes=[LATENT_SIZE * 2, LATENT_SIZE * 2, LATENT_SIZE * 2, LATENT_SIZE * 2])

lr = 1e-3
optimizer = optim.Adam(mapper.parameters(), lr=lr)

train(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=1000, optimizer=optimizer)
