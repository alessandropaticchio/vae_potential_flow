from PIL import Image
from torchvision.transforms import ToTensor
from constants import *
from utils import PicsDataset
from tqdm import tqdm
import torch.nn.functional as F

MAX_PICS = 1000

downsample = True
target_size = 100

img_path = DATA_ROOT + "fake_data/"

rays_train_set = torch.empty((1, 3, target_size, target_size))
rays_test_set = torch.empty((1, 3, target_size, target_size))
potential_train_set = torch.empty((1, 3, target_size, target_size))
potential_test_set = torch.empty((1, 3, target_size, target_size))

for i in tqdm(range(1, int(MAX_PICS * 0.8)), desc='Rays training...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    image = Image.open(img_path + 'rays_pic_data/rays_' + i + '.jpg')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

    if downsample == True:
        image = F.interpolate(image, size=(target_size, target_size))

    rays_train_set = torch.cat((rays_train_set, image), 0)

for i in tqdm(range(int(MAX_PICS * 0.8), MAX_PICS), desc='Rays test...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    image = Image.open(img_path + 'rays_pic_data/rays_' + i + '.jpg')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

    if downsample == True:
        image = F.interpolate(image, size=(target_size, target_size))

    rays_test_set = torch.cat((rays_test_set, image), 0)

for i in tqdm(range(1, int(MAX_PICS * 0.8)), desc='Potential training...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    image = Image.open(img_path + 'potential_pic_data/potential_' + i + '.jpg')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
    # Dropping alpha channel
    image = image[:, :3, :, :]

    if downsample == True:
        image = F.interpolate(image, size=(target_size, target_size))

    potential_train_set = torch.cat((potential_train_set, image), 0)

for i in tqdm(range(int(MAX_PICS * 0.8), MAX_PICS), desc='Potential test...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    image = Image.open(img_path + 'potential_pic_data/potential_' + i + '.jpg')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
    # Dropping alpha channel
    image = image[:, :3, :, :]

    if downsample == True:
        image = F.interpolate(image, size=(target_size, target_size))

    potential_test_set = torch.cat((potential_test_set, image), 0)

#  First tensor is meaningless
rays_train_set = rays_test_set[1:]
rays_test_set = rays_test_set[1:]
potential_train_set = potential_train_set[1:]
potential_test_set = potential_test_set[1:]

rays_train_set = PicsDataset(x=rays_train_set)
rays_test_set = PicsDataset(x=rays_test_set)
potential_train_set = PicsDataset(x=potential_train_set)
potential_test_set = PicsDataset(x=potential_test_set)

torch.save(rays_train_set, DATA_ROOT + '/fake_data/rays_pic_data/training_rays.pt')
torch.save(rays_test_set, DATA_ROOT + '/fake_data/rays_pic_data/test_rays.pt')
torch.save(potential_train_set, DATA_ROOT + '/fake_data/potential_pic_data/training_potential.pt')
torch.save(potential_test_set, DATA_ROOT + '/fake_data/potential_pic_data/test_potential.pt')
