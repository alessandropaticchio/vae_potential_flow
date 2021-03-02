import os
from PIL import Image
from torchvision.transforms import ToTensor
from constants import *
from tqdm import tqdm
import torch.nn.functional as F

MAX_PICS = 200

downsample = True

img_path = DATA_ROOT + "DATA21.2.18/"

rays_train_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
rays_test_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
potential_train_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_test_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

for label in tqdm(os.listdir(img_path)):
    label_name = os.fsdecode(label)
    try:
        potential = float(label_name.split('D')[1])
    except:
        continue
    if potential != 0.3:
        continue
    path = img_path + label_name
    for i in tqdm(range(1, int(MAX_PICS * 0.8) + 1), desc='Rays training...'):
        if i <= 9:
            i = '00' + str(i)
        elif 9 < i <= 99:
            i = '0' + str(i)
        else:
            i = str(i)
        image = Image.open(path + '/rays_' + i + '.jpg')
        image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

        if downsample:
            image = F.interpolate(image, size=(RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))

        rays_train_set = torch.cat((rays_train_set, image), 0)

    for i in tqdm(range(int(MAX_PICS * 0.8) + 1, MAX_PICS + 1), desc='Rays test...'):
        if i <= 9:
            i = '00' + str(i)
        elif 9 < i <= 99:
            i = '0' + str(i)
        else:
            i = str(i)
        image = Image.open(path + '/rays_' + i + '.jpg')
        image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

        if downsample:
            image = F.interpolate(image, size=(RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))

        rays_test_set = torch.cat((rays_test_set, image), 0)

    for i in tqdm(range(1, int(MAX_PICS * 0.8) + 1), desc='Potential training...'):
        if i <= 9:
            i = '00' + str(i)
        elif 9 < i <= 99:
            i = '0' + str(i)
        else:
            i = str(i)
        image = Image.open(path + '/ptnl_' + i + '.jpg')
        image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
        # image = potential * image[:, :3, :, :]
        image =  image[:, :3, :, :]

        if downsample:
            image = F.interpolate(image, size=(POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

        potential_train_set = torch.cat((potential_train_set, image), 0)

    for i in tqdm(range(int(MAX_PICS * 0.8) + 1, MAX_PICS + 1), desc='Potential test...'):
        if i <= 9:
            i = '00' + str(i)
        elif 9 < i <= 99:
            i = '0' + str(i)
        else:
            i = str(i)
        image = Image.open(path + '/ptnl_' + i + '.jpg')
        image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
        # image = potential * image[:, :3, :, :]
        image =  image[:, :3, :, :]

        if downsample:
            image = F.interpolate(image, size=(POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

        potential_test_set = torch.cat((potential_test_set, image), 0)

#  First tensor is meaningless
rays_train_set = rays_train_set[1:]
rays_test_set = rays_test_set[1:]
potential_train_set = potential_train_set[1:]
potential_test_set = potential_test_set[1:]

torch.save(rays_train_set, img_path + 'loaded_data/training_rays.pt')
torch.save(rays_test_set, img_path + 'loaded_data/test_rays.pt')
torch.save(potential_train_set, img_path + 'loaded_data/training_potential.pt')
torch.save(potential_test_set, img_path + 'loaded_data/test_potential.pt')
