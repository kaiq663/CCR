'''
We observe slightly better performance with training inputs in [0, 255] range than that in [0, 1],
so we follow AP-BSN that do not normalize the input image from [0, 255] to [0, 1].
'''
from dataset.base import BaseTrainDataset, dataset_path
import glob
import numpy as np
import os
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset
import torch

sidd_path = os.path.join(dataset_path, 'SIDD_DND/SIDD')


class SIDDSrgbTrainDataset(BaseTrainDataset):
    def __init__(self, patch_size, pin_memory):
        super(SIDDSrgbTrainDataset, self).__init__(sidd_path, patch_size, pin_memory)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
            img_H = self.imgs[index]['H']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        img_L, img_H = self.crop(img_L, img_H)
        img_L, img_H = self.augment(img_L, img_H)

        img_L, img_H = np.float32(img_L), np.float32(img_H)
        return {'L': img_L, 'H': img_H}

    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'SIDD_Medium_Srgb/Data/*/*_NOISY_SRGB_*.PNG')
        L_paths = sorted(glob.glob(L_pattern))
        for L_path in L_paths:
            self.img_paths.append({'L': L_path, 'H': L_path.replace('NOISY', 'GT')})

    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            self.imgs.append({'L': img_L, 'H': img_H})

    def _open_image(self, path):
        img = Image.open(path)
        img_np = np.asarray(img)
        img.close()
        img_np = np.transpose(img_np, (2, 0, 1))
        return img_np


class SIDDSrgbValidationDataset(Dataset):
    def __init__(self):
        super(SIDDSrgbValidationDataset, self).__init__()
        self._open_images(sidd_path)
        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_H = self.gt_block[index_n, index_k]
        img_H = np.float32(img_H)
        img_H = np.transpose(img_H, (2, 0, 1))

        img_L = self.noisy_block[index_n, index_k]
        img_L = np.float32(img_L)
        img_L = np.transpose(img_L, (2, 0, 1))

        return {'H': img_H, 'L': img_L}

    def __len__(self):
        return self.n * self.k

    def _open_images(self, path):
        mat = sio.loadmat(os.path.join(path, 'ValidationNoisyBlocksSrgb.mat'))
        self.noisy_block = mat['ValidationNoisyBlocksSrgb']
        mat = sio.loadmat(os.path.join(path, 'ValidationGtBlocksSrgb.mat'))
        self.gt_block = mat['ValidationGtBlocksSrgb']


class SIDDSrgbBenchmarkDataset(Dataset):
    def __init__(self):
        super(SIDDSrgbBenchmarkDataset, self).__init__()
        self._open_images(sidd_path)
        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_L = self.noisy_block[index_n, index_k]
        img_L = np.float32(img_L)
        img_L = np.transpose(img_L, (2, 0, 1))

        return {'L': img_L}

    def __len__(self):
        return self.n * self.k

    def _open_images(self, path):
        mat = sio.loadmat(os.path.join(path, 'BenchmarkNoisyBlocksSrgb.mat'))
        self.noisy_block = mat['BenchmarkNoisyBlocksSrgb']


dc_path = '/home/gxust/myprojects/dataset/OOD'


def open_image_SIDD(path):
    img = Image.open(path)
    img_np = np.asarray(img)
    img.close()
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

class DCValidationDataset(Dataset):
    def __init__(self, device, pin_memory):
        super().__init__()
        self.device = device
        self._img_paths = self._get_img_paths(device)
        self.pin_memory = pin_memory
        if self.pin_memory:
            self._imgs = self._open_images()

    def __getitem__(self, index):
        index = index % len(self._img_paths)

        if self.pin_memory:
            img_L = self._imgs[index]['L']
            img_H = self._imgs[index]['H']
        else:
            img_path = self._img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        # img_L, img_H = np.float32(img_L) / 255., np.float32(img_H) / 255.
        img_L, img_H = np.float32(img_L), np.float32(img_H)
        img_L, img_H = torch.from_numpy(img_L), torch.from_numpy(img_H)

        return {'L': img_L, 'H': img_H}

    def __len__(self):
        length = len(self._img_paths)
        return length

    def _get_img_paths(self, device):
        path = os.path.join(dc_path, device)
        self.img_paths = []
        if device == 'CC15':
            L_pattern = os.path.join(path, 'noisy_img/*.png')
            L_paths = sorted(glob.glob(L_pattern))
            for L_path in L_paths:
                self.img_paths.append(
                    {'L': L_path, 'H': L_path.replace('noisy_img', 'clean_img').replace('real.png', 'mean.png')})
        elif device == 'PolyU':
            L_pattern = os.path.join(path, 'noisy_img/*.JPG')
            L_paths = sorted(glob.glob(L_pattern))
            for L_path in L_paths:
                self.img_paths.append(
                    {'L': L_path, 'H': L_path.replace('noisy_img', 'clean_img').replace('real.JPG', 'mean.JPG')})
        else:
            L_pattern = os.path.join(path, 'noisy_img/*.png')
            L_paths = sorted(glob.glob(L_pattern))
            for L_path in L_paths:
                self.img_paths.append(
                    {'L': L_path, 'H': L_path.replace('noisy_img', 'clean_img').replace('noisy.png', 'clean.png')})
        return self.img_paths

    def _open_images(self):
        imgs = []
        for img_path in self._img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            imgs.append({'L': img_L, 'H': img_H})
        return imgs

    def _open_image(self, path):
        return open_image_SIDD(path)
