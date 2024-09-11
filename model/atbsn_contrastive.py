from model.base import BaseModel
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        model = models.vgg19(pretrained=False)
        model.load_state_dict(torch.load('../pretrained_models/vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ATBSN_Contrastive_Model(BaseModel):
    def __init__(self, opt):
        super(ATBSN_Contrastive_Model, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_BNN = torch.optim.Adam(self.networks['BNN'].parameters(), lr=opt['lr'])
        self.scheduler_BNN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_BNN, opt['BNN_iters'])
        self.optimizer_UNet = torch.optim.Adam(self.networks['UNet'].parameters(), lr=opt['lr'])
        self.scheduler_UNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_UNet, opt['UNet_iters'])
        self.vgg = Vgg19().cuda()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def criterion_distill(self, unet, bnn0, bnn1, bnn3, bnn5, bnn7, bnn9, bnn11=None, bnn13=None, bnn15=None, type='multi'):
        loss = 0.
        if type == "mean":
            mean = bnn0 + bnn1 + bnn3 + bnn5 + bnn7 + bnn9
            mean /= 6.
            loss = self.criteron(mean, unet)

        if type == "multi":
            loss += self.criteron(bnn0, unet)
            loss += self.criteron(bnn1, unet)
            loss += self.criteron(bnn3, unet)
            loss += self.criteron(bnn5, unet)
            loss += self.criteron(bnn7, unet)
            loss += self.criteron(bnn9, unet)
            # loss += self.criteron(bnn11, unet)
            # loss += self.criteron(bnn13, unet)
            # loss += self.criteron(bnn15, unet)

        return loss

    def criterion_contrastive(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)

        loss = 0.
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.criteron(a_vgg[i], p_vgg[i].detach())
            d_an = self.criteron(a_vgg[i], n_vgg[i].detach())
            contrastive = d_ap / (d_an + 1e-7)
            loss += self.weights[i] * contrastive

        return loss

    def train_step(self, data, output_ood_2, output_ood_1):
        self.iter += 1
        self.update_stage()

        input = data['L']
        if self.stage == 'BNN':
            self.networks['BNN'].train()
            bnn = self.networks['BNN'](input)
            self.loss = self.criteron(bnn, input)
            self.optimizer_BNN.zero_grad()
            self.loss.backward()
            self.optimizer_BNN.step()
            self.scheduler_BNN.step()

        elif self.stage == 'UNet':
            self.optimizer_UNet.zero_grad()

            self.networks['UNet'].train()
            unet = self.networks['UNet'](input)
            self.loss_l1 = self.criteron(unet, output_ood_2) + self.criteron(unet, output_ood_1)
            self.loss_cr = self.criterion_contrastive(unet, output_ood_2, input)
            self.loss = self.loss_l1 + self.loss_cr

            self.loss.backward()
            self.optimizer_UNet.step()
            self.scheduler_UNet.step()

    def validation_step(self, data):
        self.update_stage()
        input = data['L']
        if self.stage == 'BNN':
            self.networks['BNN'].eval()
            with torch.no_grad():
                output = self.networks['BNN'](input)
        elif self.stage == 'UNet':
            self.networks['UNet'].eval()
            with torch.no_grad():
                output = self.networks['UNet'](input)

        return output

    def save_net(self):
        if self.stage == 'BNN':
            net = self.networks['BNN']
        elif self.stage == 'UNet':
            net = self.networks['UNet']

        if isinstance(net, DataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], 'net_iter_%08d.pth' % self.iter))

    def save_model(self):
        if self.stage == 'BNN':
            save_dict = {'iter': self.iter,
                         'optimizer_BNN': self.optimizer_BNN.state_dict(),
                         'scheduler_BNN': self.scheduler_BNN.state_dict(),
                         'BNN': self.networks['BNN'].state_dict()}
        elif self.stage == 'UNet':
            save_dict = {'iter': self.iter,
                         'optimizer_UNet': self.optimizer_UNet.state_dict(),
                         'scheduler_UNet': self.scheduler_UNet.state_dict(),
                         'BNN': self.networks['BNN'].state_dict(),
                         'UNet': self.networks['UNet'].state_dict()}
        torch.save(save_dict, os.path.join(self.opt['log_dir'], 'model_iter_%08d.pth' % self.iter))

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict['iter']
        self.update_stage()
        if self.stage == 'BNN':
            self.optimizer_BNN.load_state_dict(load_dict['optimizer_BNN'])
            self.scheduler_BNN.load_state_dict(load_dict['scheduler_BNN'])
            self.networks['BNN'].load_state_dict(load_dict['BNN'])
        elif self.stage == 'UNet':
            self.optimizer_UNet.load_state_dict(load_dict['optimizer_UNet'])
            self.scheduler_UNet.load_state_dict(load_dict['scheduler_UNet'])
            self.networks['BNN'].load_state_dict(load_dict['BNN'])
            self.networks['UNet'].load_state_dict(load_dict['UNet'])
        else:
            raise NotImplementedError

    def update_stage(self):
        if self.iter <= self.opt['BNN_iters']:
            self.stage = 'BNN'
        else:
            self.stage = 'UNet'
