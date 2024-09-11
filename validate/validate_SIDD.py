import sys
sys.path.append('..')
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.utils.data import DataLoader
from utils.option import parse, recursive_print
from utils.build import build

from tqdm import tqdm

def validate_sidd(model, sidd_loader):
    psnrs, count = 0, 0
    for data in tqdm(sidd_loader):
        output = model.validation_step(data)
        output = torch.floor(output + 0.5)
        output = torch.clamp(output, 0, 255)
        output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=255)
        psnrs += psnr
        count += 1
    return psnrs / count

def validate_sidd_tbsn(model, sidd_loader):
    psnrs, ssims, count = 0, 0, 0
    for data in tqdm(sidd_loader):
        data['H'] /= 255.
        output = model.validation_step(data)
        output = torch.floor(output + 0.5) / 255.
        output = torch.clamp(output, 0, 1)
        output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()

        psnr = peak_signal_noise_ratio(output, gt, data_range=1)
        ssim = structural_similarity(output, gt, data_range=1, gaussian_weights=True, channel_axis=2, use_sample_covariance=False)

        ssims += ssim
        psnrs += psnr

        count += 1
    return psnrs/count, ssims/count


def main(opt):
    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()

    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])

    message = ''
    message_psnr = ''
    message_ssim = ''
    psnrs, ssims, count = 0., 0., 0

    # for validation_loader in validation_loaders:
    #     if validation_loader.dataset.__class__.__name__ == 'SIDDSrgbValidationDataset':
    #         psnr = validate_sidd(model, validation_loader)
    #         message += '%s: %6.4f  ' % ("SIDD", psnr)
    #     else:
    #         psnr = validate_sidd(model, validation_loader)
    #         message += '%6.4f ' % (psnr)
    #         psnrs += psnr

    # if len(validation_loaders)>1:
    #     message += '%s: %6.4f' % ('OOD_avg', psnrs/(len(validation_loaders)-1))
    # print(message)

    for validation_loader in validation_loaders:
        if validation_loader.dataset.__class__.__name__ == 'SIDDSrgbValidationDataset':
            psnr, ssim = validate_sidd_tbsn(model, validation_loader)
            message_psnr += '%s: %6.4f  ' % ("SIDD", psnr)
            message_ssim += '%s: %6.4f  ' % ("SIDD", ssim)

        else:
            psnr, ssim = validate_sidd_tbsn(model, validation_loader)
            message_psnr += '%6.4f ' % (psnr)
            message_ssim += '%6.4f ' % (ssim)
            psnrs += psnr
            ssims += ssim

    if len(validation_loaders)>1:
        message_psnr += '%s: %6.4f' % ('OOD_avg_psnr', psnrs/(len(validation_loaders)-1))
        message_ssim += '%s: %6.4f' % ('OOD_avg_ssim', ssims/(len(validation_loaders)-1))

    print(message_psnr)
    print(message_ssim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/atbsn_lite.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    recursive_print(opt)

    main(opt)