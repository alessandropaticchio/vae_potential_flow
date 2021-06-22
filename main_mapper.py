from constants import *
from models import Mapper
from training import train
import torch.optim as optim

data_path = DATA_ROOT + 'RP_images/mapped/'

train_dataset = torch.load(data_path + 'training.pt')
test_dataset = torch.load(data_path + 'test.pt')

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

h_sizes = H_SIZES

mapper = Mapper(h_sizes=h_sizes)

lr = 1e-3
optimizer = optim.Adam(mapper.parameters(), lr=lr, weight_decay=0.1)

train(net=mapper, train_loader=train_loader, test_loader=test_loader, epochs=1000, optimizer=optimizer, early_stopping=True)
