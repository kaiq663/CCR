from model.base import BaseModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, train=False):
    b,c,w,h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    unshuffled = unshuffled.view(b, c, f * f, w // f, h // f).permute(0, 2, 1, 3, 4)
    if train:
        unshuffled = unshuffled[:, 0]
        unshuffled = unshuffled.reshape(b, c, w // f, h // f)
    else:
        unshuffled = unshuffled.reshape(b * f * f, c, w // f, h // f)
    return unshuffled

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int):
    b,c,h,w = x.shape
    before_shuffle = x.view(b // (f * f), f * f, c, h, w).permute(0, 2, 1, 3, 4)
    before_shuffle = before_shuffle.reshape(b // (f * f), c * f * f, h, w)
    return F.pixel_shuffle(before_shuffle, f)

class APBSN_TBSN_Lite_Model(BaseModel):
    def __init__(self, opt):
        super(APBSN_TBSN_Lite_Model, self).__init__(opt)
        self.pd_a = opt['pd_a']
        self.pd_b = opt['pd_b']
        self.R3 = opt['R3']
        self.R3_T = opt['R3_T']
        self.R3_p = opt['R3_p']


    def validation_step(self, data):
        input = data['L'].to(self.device)
        b, c, h, w = input.shape
        input_pd = pixel_shuffle_down_sampling(input, f=self.pd_b)

        self.networks['bsn'].eval()
        with torch.no_grad():
            output_pd = self.networks['bsn'](input_pd)

        output = pixel_shuffle_up_sampling(output_pd, f=self.pd_b)
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
                    tmp_output = self.networks['bsn'](tmp_input)
                denoised[..., t] = tmp_output

            denoised = torch.mean(denoised, dim=-1)

        return denoised

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
                state_dict[key.replace('bsn.', '')] =  state_dict.pop(key)
        net.load_state_dict(state_dict)
