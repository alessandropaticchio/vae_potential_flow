from training import train_ae
from models import ConvPlainAE
from constants import *
from utils import StrengthDataset, generate_dataset_from_strength
import torch
import torch.optim as optim

batch_size = 32

dataset = 'total'
vae_type = 'conv'
dataset = 'potential'

strengths = STRENGTHS

pics_train_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_' + dataset + '.pt')
pics_test_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_' + dataset + '.pt')
strength_train_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'training_strength.pt')
strength_test_dataset = torch.load(DATA_ROOT + 'num=999_unscaled/loaded_data/' + 'test_strength.pt')

pics_train_dataset, strength_train_dataset = generate_dataset_from_strength(pics_train_dataset, strength_train_dataset,
                                                                            strengths)
pics_test_dataset, strength_test_dataset = generate_dataset_from_strength(pics_test_dataset, strength_test_dataset,
                                                                          strengths)

train_dataset = StrengthDataset(x=pics_train_dataset, d=strength_train_dataset)
test_dataset = StrengthDataset(x=pics_test_dataset, d=strength_test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

ae = ConvPlainAE(image_channels=3)

lr = 1e-3
optimizer = optim.Adam(ae.parameters(), lr=lr)

train_ae(net=ae, train_loader=train_loader, test_loader=test_loader, epochs=600, optimizer=optimizer)
