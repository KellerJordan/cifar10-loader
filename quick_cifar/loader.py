## https://github.com/KellerJordan/cifar10-dataloader/blob/master/quick_cifar/loader.py
import os
from math import ceil
import torch
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py#L389
def make_random_square_masks(inputs, mask_size):
    assert mask_size > 0
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    # seed centers of squares to cutout boxes from, in one dimension each
    mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-2]-mask_size//2-is_even)
    mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-1]-mask_size//2-is_even)

    # measure distance, using the center as a reference point
    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)

    final_mask = to_mask_y * to_mask_x

    return final_mask

def batch_flip_lr(batch_images, flip_chance=.5):
    return torch.where(torch.rand_like(batch_images[:, 0, 0, 0]).view(-1, 1, 1, 1) < flip_chance,
                       batch_images.flip(-1), batch_images)

def batch_crop(inputs, crop_size):
    crop_mask_batch = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask_batch)
    return cropped_batch.view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)

def batch_translate(inputs, translate):
    width = inputs.shape[2]
    inputs = F.pad(inputs, (translate,)*4, 'constant', value=0)
    return batch_crop(inputs, width)

def batch_cutout(inputs, size):
    masks = make_random_square_masks(inputs, size)
    cutout_batch = inputs.masked_fill(masks, 0)
    return cutout_batch

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, keep_last=False):
        dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
        imgs = torch.tensor(dset.data).cuda()
        imgs = (imgs.float() / 255).permute(0, 3, 1, 2)
        self.mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).cuda()
        self.images = (imgs - self.mean) / self.std
        self.targets = torch.tensor(dset.targets).cuda()
        
        # set defaults
        self.aug = {'flip': False, 'translate': 0, 'cutout': 0, **(aug or {})}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.keep_last = keep_last

    def __len__(self):
        return ceil(len(self.images)/self.batch_size) if self.keep_last else len(self.images)//self.batch_size

    def __iter__(self):
        images = self.images
        targets = self.targets

        if self.aug['flip']:
            images = batch_flip_lr(images)
        if self.aug['cutout'] > 0:
            images = batch_cutout(images, self.aug['cutout'])
        if self.aug['translate'] > 0:
            images = batch_translate(images, self.aug['translate'])

        images = images.to(memory_format=torch.channels_last)
        
        shuffled = torch.randperm(len(images), device=images.device)
        for i in range(len(self)):
            idxs = shuffled[i*self.batch_size:(i+1)*self.batch_size]
            yield (images.index_select(0, idxs), targets.index_select(0, idxs))

