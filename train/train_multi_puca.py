import argparse
import torch
from torch.utils.data import DataLoader
from utils.build import build
from utils.io import log
from utils.option import parse, recursive_log
from validate.validate_SIDD import validate_sidd

def main(opt, opt_ood):
    train_loaders = []
    for train_dataset_opt in opt['train_datasets']:
        TrainDataset = getattr(__import__('dataset'), train_dataset_opt['type'])
        train_set = build(TrainDataset, train_dataset_opt['args'])
        train_loader = DataLoader(train_set, batch_size=train_dataset_opt['batch_size'], shuffle=False,
                                  num_workers=train_dataset_opt['batch_size'], drop_last=True)
        train_loaders.append(train_loader)

    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()

    Model = getattr(__import__('model'), opt_ood['model'])
    model_ood = Model(opt_ood)
    model_ood.data_parallel()

    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])

    def train_step(data):
        # output_ood_4 = model_ood.validation_step(data, r3_options=[8,0.16,4])
        output_ood_2 = model_ood.validation_step(data, r3_options=[8, 0.16, 2])
        output_ood_1 = model_ood.validation_step(data, r3_options=[8, 0.16 ,1])

        model.train_step(data, output_ood_2, output_ood_1)

        if model.iter % opt['print_every'] == 0:
            model.log()

        if model.iter % opt['save_every'] == 0:
            model.save_net()
            #model.save_model()

        if model.iter % opt['validate_every'] == 0:
            message = 'iter: %d, ' % model.iter
            psnrs = 0
            for validation_loader in validation_loaders:
                if validation_loader.dataset.__class__.__name__ == 'SIDDSrgbValidationDataset':
                    psnr = validate_sidd(model, validation_loader)
                    message += '%s: %6.4f  ' % ('SIDD', psnr)
                else:
                    psnr = validate_sidd(model, validation_loader)
                    message += '%6.4f  ' % (psnr)
                    psnrs += psnr
            message += '%s: %6.4f' % ('OOD_avg', psnrs/5.)
            log(opt['log_file'], message + '\n')

        if model.iter == opt['num_iters']:
            model.save_net()
            exit()

    while True:
        for data in train_loaders[0]:
            data = {'L': data['L'].cuda(), 'H': data['H'].cuda()} if 'H' in data else {'L': data['L'].cuda()}
            train_step(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")

    parser.add_argument("--config", type=str, default='option/atbsn_c2pnet.json')
    parser.add_argument("--config2", type=str, default='option/apbsn_puca_lite.json')

    argspar = parser.parse_args()

    opt = parse(argspar.config)
    opt_ood = parse(argspar.config2)
    recursive_log(opt['log_file'], opt)
    recursive_log('/home/gxust/myprojects/SASSL/logs/log.out', opt)
    main(opt, opt_ood)