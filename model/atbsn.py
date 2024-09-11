from model.base import BaseModel
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel


class ATBSNModel(BaseModel):
    def __init__(self, opt):
        super(ATBSNModel, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_BNN = torch.optim.Adam(self.networks['BNN'].parameters(), lr=opt['lr'])
        self.scheduler_BNN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_BNN, opt['BNN_iters'])
        self.optimizer_UNet = torch.optim.Adam(self.networks['UNet'].parameters(), lr=opt['lr'])
        self.scheduler_UNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_UNet, opt['UNet_iters'])

    def train_step(self, data,):
        self.iter += 1
        self.update_stage()

        input = data['L']


        if self.stage == 'BNN':
            self.networks['BNN'].train()
            BNN = self.networks['BNN'](input)
            self.loss = self.criteron(BNN, input)
            self.optimizer_BNN.zero_grad()
            self.loss.backward()
            self.optimizer_BNN.step()
            self.scheduler_BNN.step()

        elif self.stage == 'UNet':
            self.networks['BNN'].eval()
            self.networks['UNet'].train()
            with torch.no_grad():
                BNN0 = self.networks['BNN'](input,shift=0)#0
                BNN1 = self.networks['BNN'](input,shift=1)#1
                BNN3 = self.networks['BNN'](input,shift=2)#3
                BNN5 = self.networks['BNN'](input,shift=3)#5
                BNN7 = self.networks['BNN'](input,shift=4)#7
                BNN9 = self.networks['BNN'](input,shift=5)#9
                # BNN11 = self.networks['BNN'](input,shift=6)#11
                # BNN13 = self.networks['BNN'](input,shift=7)#13
                # BNN15 = self.networks['BNN'](input,shift=8)#15

            UNet = self.networks['UNet'](input)

            self.loss = 0.
            self.loss += self.criteron(BNN0, UNet)
            self.loss += self.criteron(BNN1, UNet)
            self.loss += self.criteron(BNN3, UNet)
            self.loss += self.criteron(BNN5, UNet)
            self.loss += self.criteron(BNN7, UNet)
            self.loss += self.criteron(BNN9, UNet)
            # self.loss += self.criteron(BNN11, UNet)
            # self.loss += self.criteron(BNN13, UNet)
            # self.loss += self.criteron(BNN15, UNet)

            self.optimizer_UNet.zero_grad()
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
