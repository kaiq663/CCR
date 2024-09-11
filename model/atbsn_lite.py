from model.base import BaseModel
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel


class ATBSN_Lite_Model(BaseModel):
    def __init__(self, opt):
        super(ATBSN_Lite_Model, self).__init__(opt)

    def validation_step(self, data):
        input = data['L']
        self.networks['UNet'].eval()
        with torch.no_grad():
            output = self.networks['UNet'](input)
        return output