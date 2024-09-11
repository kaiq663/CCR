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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)
def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2
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
            noisy1, noisy2 = pair_downsampler(inp)
            pred1 = self.model_restoration(noisy1)
            pred2 = self.model_restoration(noisy2)
            noisy_denoised = self.model_restoration(inp)
            denoised1, denoised2 = pair_downsampler(noisy_denoised)
            return noisy1, noisy2, pred1, pred2, denoised1, denoised2
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
            noisy1, noisy2, pred1, pred2, denoised1, denoised2 = model.forward_lan(noisy_torch, is_train=True)
            loss_res =  0.5*  (mse(noisy1, pred2) + mse(noisy2, pred1))
            loss_cons = 0.5* (mse(pred1, denoised1) + mse(pred2, denoised2))
            loss = loss_res + loss_cons
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
