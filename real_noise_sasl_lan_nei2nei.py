import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from mash.imlib import imlib
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from network.unet import UNet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    PSNR = 0
    # SDAP
    # Img = img_as_ubyte(Img)
    # Iclean = img_as_ubyte(Iclean)
    # for i in range(Img.shape[0]):
    #     PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=255)

    # PUCA
    Img = np.clip(Img, 0, 255)
    Iclean = np.clip(Iclean, 0, 255)
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=255)
    return (PSNR / Img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    SSIM = 0
    # SDAP
    # Img = img_as_ubyte(Img)
    # Iclean = img_as_ubyte(Iclean)
    # for i in range(Img.shape[0]):
    #     SSIM += structural_similarity(Iclean[i,:,:,:].transpose((1,2,0)), Img[i,:,:,:].transpose((1,2,0)),
    #                          data_range=255, gaussian_weights=True,
    #                                   channel_axis=2, use_sample_covariance=False)

    # PUCA
    Img = np.clip(Img, 0, 255)
    Iclean = np.clip(Iclean, 0, 255)
    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i, :, :, :].transpose((1, 2, 0)), Img[i, :, :, :].transpose((1, 2, 0)),
                                      data_range=255, multichannel=True, channel_axis=2)
    return (SSIM / Img.shape[0])

class LAN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_restoration = UNet()
        checkpoint = torch.load(args.weights)
        self.model_restoration.load_state_dict(checkpoint)

        self.adapted = nn.Parameter(torch.zeros(1,3,256,256), requires_grad=True)

    def forward_fully_trainable(self, inp, is_train=False):
        if is_train:
            noisy1, noisy2 = pair_downsampler(inp)
            pred1 = self.model_restoration(noisy1)
            pred2 = self.model_restoration(noisy2)
            noisy_denoised = self.model_restoration(inp)
            denoised1, denoised2 = pair_downsampler(noisy_denoised)
            return noisy1, noisy2, pred1, pred2, denoised1, denoised2
        else:
            out = self.model_restoration(inp)
            return out
    def forward_lan(self, inp, is_train=False):
        inp = inp + self.adapted
        if is_train:
            for p in self.model_restoration.parameters():
                p.requires_grad = False
            mask1, mask2 = generate_mask_pair(inp)
            noisy_sub1 = generate_subimages(inp, mask1)
            noisy_sub2 = generate_subimages(inp, mask2)
            noisy_denoised = self.model_restoration(inp)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
            noisy_output = self.model_restoration(noisy_sub1)
            noisy_target = noisy_sub2
            return noisy_output, noisy_target, noisy_sub1_denoised, noisy_sub2_denoised
        else:
            out = self.model_restoration(inp)
            return out

def main(args):
    mobile = 'IPHONE'
    # noisy = '../dataset/OOD/{mobile}/noisy_img'.format(mobile=mobile)
    # clean = '../dataset/OOD/{mobile}/clean_img'.format(mobile=mobile)
    noisy = '../dataset/AFM/{mobile}/input_crops'.format(mobile=mobile)
    clean = '../dataset/AFM/{mobile}/gt_crops'.format(mobile=mobile)
    paths_noisy = sorted(os.listdir(noisy))
    paths_clean = sorted(os.listdir(clean))
    rgb_chw_cv2 = imlib('rgb', fmt='chw', lib='cv2')

    psnrs = 0.0
    ssims = 0.0
    iters = 20
    for i in range(len(paths_noisy)):
        noisy_orig_np = np.float32(
            rgb_chw_cv2.read(os.path.join(noisy, paths_noisy[i])))
        clean_orig_np = np.float32(
            rgb_chw_cv2.read(os.path.join(clean, paths_clean[i])))
        noisy_torch = torch.from_numpy(noisy_orig_np).unsqueeze(0).to(device)
        clean_torch = torch.from_numpy(clean_orig_np).unsqueeze(0).to(device)

        model = LAN(args).to(device)
        model.train()

        optimizer_lan = torch.optim.Adam(model.parameters(), lr=5e-2, betas=(0.9, 0.999))

        for iter in range(iters):
            noisy_output,noisy_target,noisy_sub1_denoised, noisy_sub2_denoised = model.forward_lan(noisy_torch, is_train=True)
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            loss1 = torch.mean(diff ** 2)
            loss2 = torch.mean((diff - exp_diff) ** 2)
            loss = loss1 + loss2
            optimizer_lan.zero_grad()
            loss.backward()
            optimizer_lan.step()

            if (iter+1)%iters == 0:
                model.eval()
                with torch.no_grad():
                    output = model.forward_lan(noisy_torch, is_train=False)
                output = torch.floor(output+0.5)
                psnr = batch_PSNR(output, clean_torch)
                #ssim = batch_SSIM(output, clean_torch)
                print(i, psnr)
                psnrs += psnr
                #ssims += ssim

    print('----------------------')
    print('----------------------')
    print('----------------------')
    print(psnrs / len(paths_noisy))
    print(ssims / len(paths_noisy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real Image denoising')
    parser.add_argument('--weights', default='pretrained_models/ood_38_77psnr.pth', type=str, help='Path to weights')
    args = parser.parse_args()
    main(args)
