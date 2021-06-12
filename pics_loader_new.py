import os
from PIL import Image
from torchvision.transforms import ToTensor
from constants import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

MAX_PICS = 2000

downsample = True

strenghts = STRENGTHS

img_path = DATA_ROOT + '/RP_images/'

rays_train_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
rays_test_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))
rays_data_set = torch.empty((1, RAYS_IMAGE_CHANNELS, RAYS_IMAGE_SIZE, RAYS_IMAGE_SIZE))

potential_train_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_test_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))
potential_data_set = torch.empty((1, POTENTIAL_IMAGE_CHANNELS, POTENTIAL_IMAGE_SIZE, POTENTIAL_IMAGE_SIZE))

strength_train = torch.empty(1, 1)
strength_test = torch.empty(1, 1)
strength_data_set = torch.empty(1, 1)

n_strenghts = 0

for label in tqdm(os.listdir(img_path)):
    label_name = os.fsdecode(label)
    try:
        potential = float(label_name.split('D')[1].split(' ')[0])
        if potential in strenghts:
            n_strenghts += 1
        else:
            continue
    except:
        continue
    path = img_path + '/' + label_name
    for i in tqdm(range(1, int(MAX_PICS) + 1), desc='Rays data strength - ' + str(potential)):
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

    for i in tqdm(range(1, int(MAX_PICS) + 1), desc='Potential data strength - ' + str(potential)):
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
        strength_data_set = torch.cat((strength_data_set, potential_tensor), 0)


#  First tensor is meaningless
rays_data_set = rays_data_set[1:]
potential_data_set = potential_data_set[1:]
strength_data_set = strength_data_set[1:]

train_index = np.random.choice(int(MAX_PICS * n_strenghts), int(0.8 * MAX_PICS * n_strenghts), replace=False)
test_index = [i for i in range(0, int(MAX_PICS * n_strenghts)) if i not in train_index]

rays_train_set = rays_data_set[train_index]
potential_train_set = potential_data_set[train_index]
strength_train = strength_data_set[train_index]

rays_test_set = rays_data_set[test_index]
potential_test_set = potential_data_set[test_index]
strength_test = strength_data_set[test_index]

torch.save(rays_train_set, img_path + 'loaded_data/training_rays.pt')
torch.save(rays_test_set, img_path + 'loaded_data/test_rays.pt')
torch.save(potential_train_set, img_path + 'loaded_data/training_potential.pt')
torch.save(potential_test_set, img_path + 'loaded_data/test_potential.pt')
torch.save(strength_train, img_path + 'loaded_data/training_strength.pt')
torch.save(strength_test, img_path + 'loaded_data/test_strength.pt')
