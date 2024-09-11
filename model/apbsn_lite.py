from model.base import BaseModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
class APBSN_Lite_Model(BaseModel):
    def __init__(self, opt):
        super(APBSN_Lite_Model, self).__init__(opt)
        self.R3 = opt['R3']
        self.R3_T = opt['R3_T']
        self.R3_p = opt['R3_p']
        self.pd_b = opt['pd_b']
        self.pd_pad = opt['pd_pad']

    def validation_step(self, data):
        input = data['L'].to(self.device)
        b, c, h, w = input.shape
        self.networks['bsn'].eval()

        pd_input = pixel_shuffle_down_sampling(input, f=self.pd_b, pad=self.pd_pad)
        with torch.no_grad():
            pd_output = self.networks['bsn'](pd_input)
        output = pixel_shuffle_up_sampling(pd_output, f=self.pd_b, pad=self.pd_pad)

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
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                with torch.no_grad():
                    tmp_output = self.networks['bsn'](tmp_input)[:, :, p:-p, p:-p]
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
                state_dict[key.replace('bsn.', '')] = state_dict.pop(key)
        net.load_state_dict(state_dict)
