from PIL import Image
from torchvision.transforms import ToTensor
from constants import *
from utils import PicsDataset
from tqdm import tqdm
import torch.nn.functional as F

MAX_PICS = 1000

downsample = True

img_path = DATA_ROOT + "real_data/"

rays_train_set = torch.empty((1, 1, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
rays_test_set = torch.empty((1, 1, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
potential_train_set = torch.empty((1, 1, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_test_set = torch.empty((1, 1, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

for i in tqdm(range(1, int(MAX_PICS * 0.8)), desc='Rays training...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    # Loading pic and passing from 3 channels to just 1
    image = Image.open(img_path + 'rays_pic_data/rays_' + i + '.jpg').convert('L')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension


    if downsample == True:
        image = F.interpolate(image, size=(RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))

    rays_train_set = torch.cat((rays_train_set, image), 0)

for i in tqdm(range(int(MAX_PICS * 0.8), MAX_PICS), desc='Rays test...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    # Loading pic and passing from 3 channels to just 1
    image = Image.open(img_path + 'rays_pic_data/rays_' + i + '.jpg').convert('L')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

    if downsample == True:
        image = F.interpolate(image, size=(RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))

    rays_test_set = torch.cat((rays_test_set, image), 0)

for i in tqdm(range(1, int(MAX_PICS * 0.8)), desc='Potential training...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    # Loading pic and passing from 3 channels to just 1
    image = Image.open(img_path + 'potential_pic_data/potential_' + i + '.jpg').convert('L')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

    if downsample == True:
        image = F.interpolate(image, size=(POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

    potential_train_set = torch.cat((potential_train_set, image), 0)

for i in tqdm(range(int(MAX_PICS * 0.8), MAX_PICS), desc='Potential test...'):
    if i <= 9:
        i = '00' + str(i)
    elif 9 < i <= 99:
        i = '0' + str(i)
    else:
        i = str(i)
    # Loading pic and passing from 3 channels to just 1
    image = Image.open(img_path + 'potential_pic_data/potential_' + i + '.jpg').convert('L')
    image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

    if downsample == True:
        image = F.interpolate(image, size=(POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

    potential_test_set = torch.cat((potential_test_set, image), 0)

#  First tensor is meaningless
rays_train_set = rays_train_set[1:]
rays_test_set = rays_test_set[1:]
potential_train_set = potential_train_set[1:]
potential_test_set = potential_test_set[1:]

rays_train_set = PicsDataset(x=rays_train_set)
rays_test_set = PicsDataset(x=rays_test_set)
potential_train_set = PicsDataset(x=potential_train_set)
potential_test_set = PicsDataset(x=potential_test_set)

torch.save(rays_train_set, img_path + 'rays_pic_data/training_rays.pt')
torch.save(rays_test_set, img_path + 'rays_pic_data/test_rays.pt')
torch.save(potential_train_set, img_path + 'potential_pic_data/training_potential.pt')
torch.save(potential_test_set, img_path + 'potential_pic_data/test_potential.pt')
