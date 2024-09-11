from model.base import BaseModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel


class APBSN_PUCA_Lite_Model(BaseModel):
    def __init__(self, opt):
        super(APBSN_PUCA_Lite_Model, self).__init__(opt)
        self.R3 = opt['R3']
        self.R3_T = opt['R3_T']
        self.R3_p = opt['R3_p']

    def validation_step(self, data, r3_options=None):
        if r3_options:
            self.R3_T = r3_options[0]
            self.R3_p = r3_options[1]
            self.kb = r3_options[2]

        input = data['L'].to(self.device)
        b, c, h, w = input.shape
        self.networks['bsn'].eval()

        with torch.no_grad():
            output = self.networks['bsn'](input, option=self.kb)

        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            denoised = output[:, :, :h, :w]
        else:
            denoised = torch.empty(*(input.shape), self.R3_T, device=input.device)
            # torch.manual_seed(0)
            for t in range(self.R3_T):
                indice = torch.rand_like(input)
                mask = indice < self.R3_p
                tmp_input = torch.clone(output).detach()
                tmp_input[mask] = input[mask]
                with torch.no_grad():
                    tmp_output = self.networks['bsn'](x=tmp_input, refine=True, option=self.kb)
                denoised[..., t] = tmp_output
            denoised = torch.mean(denoised, dim=-1)
        return denoised

    # def validation_step(self, data):
    #     input = data['L'].to(self.device)
    #     b, c, h, w = input.shape
    #     self.networks['bsn'].eval()
    #
    #     with torch.no_grad():
    #         output = self.networks['bsn'](input)
    #
    #     if not self.R3:
    #         ''' Directly return the result (w/o R3) '''
    #         denoised = output[:, :, :h, :w]
    #     else:
    #         denoised = torch.empty(*(input.shape), self.R3_T, device=input.device)
    #         # torch.manual_seed(0)
    #         for t in range(self.R3_T):
    #             indice = torch.rand_like(input)
    #             mask = indice < self.R3_p
    #             tmp_input = torch.clone(output).detach()
    #             tmp_input[mask] = input[mask]
    #             with torch.no_grad():
    #                 tmp_output = self.networks['bsn'](x=tmp_input, refine=True)
    #             denoised[..., t] = tmp_output
    #         denoised = torch.mean(denoised, dim=-1)
    #     return denoised

    def data_parallel(self):
        self.device = torch.device('cuda')
        for name in self.networks.keys():
            net = self.networks[name]
            net = net.cuda()
            net = DataParallel(net)
            self.networks[name] = net

    def load_net(self, net, path):
        state_dict = torch.load(path, map_location='cpu')
        if 'model_weight' in state_dict:
            state_dict = state_dict['model_weight']['denoiser']
        if 'bsn' in list(state_dict.keys())[0]:
            for key in list(state_dict.keys()):
                state_dict[key.replace('bsn.', '')] = state_dict.pop(key)
        net.load_state_dict(state_dict)