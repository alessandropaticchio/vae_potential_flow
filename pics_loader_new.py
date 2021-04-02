import os
from PIL import Image
from torchvision.transforms import ToTensor
from constants import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

MAX_PICS = 200

downsample = True

img_path = DATA_ROOT

rays_train_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
rays_test_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
rays_data_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
potential_train_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_test_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_data_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_strength_train = torch.empty(1, 1)
potential_strength_test = torch.empty(1, 1)
potential_strength_data = torch.empty(1, 1)

for label in tqdm(os.listdir(img_path)):
    label_name = os.fsdecode(label)
    try:
        potential = float(label_name.split('D')[1].split(' ')[0])
    except:
        continue
    if potential != 0.3:
        continue
    path = img_path + label_name
    for i in tqdm(range(1, int(MAX_PICS) + 1), desc='Rays data...'):
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

        rays_data_set = torch.cat((rays_data_set, image), 0)

    for i in tqdm(range(1, int(MAX_PICS) + 1), desc='Potential data...'):
        if i <= 9:
            i = '00' + str(i)
        elif 9 < i <= 99:
            i = '0' + str(i)
        else:
            i = str(i)
        image = Image.open(path + '/ptnl_' + i + '.jpg')
        image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
        # image = potential * image[:, :3, :, :]
        image = image[:, :3, :, :]

        if downsample:
            image = F.interpolate(image, size=(POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

        potential_data_set = torch.cat((potential_data_set, image), 0)
        potential_tensor = torch.tensor([[[potential]]]).squeeze(0)
        potential_strength_data = torch.cat((potential_strength_data, potential_tensor), 0)


#  First tensor is meaningless
rays_data_set = rays_data_set[1:]
potential_data_set = potential_data_set[1:]
potential_strength_data = potential_strength_data[1:]

index = np.random.choice(MAX_PICS, int(0.8 * MAX_PICS), replace=False)

rays_train_set = rays_data_set[index]
potential_train_set = potential_data_set[index]
potential_strength_train = potential_strength_data[index]

rays_test_set = np.delete(rays_data_set, index)
potential_test_set = np.delete(potential_data_set, index)
potential_strength_test = np.delete(potential_strength_data, index)

torch.save(rays_train_set, img_path + 'loaded_data/training_rays.pt')
torch.save(rays_test_set, img_path + 'loaded_data/test_rays.pt')
torch.save(potential_train_set, img_path + 'loaded_data/training_potential.pt')
torch.save(potential_test_set, img_path + 'loaded_data/test_potential.pt')
torch.save(potential_strength_train, img_path + 'loaded_data/train_strength_potential.pt')
torch.save(potential_strength_test, img_path + 'loaded_data/test_strength_potential.pt')
