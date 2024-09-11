import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
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
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,
                                                                                                           w + 2 * f * pad,
                                                                                                           h + 2 * f * pad)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c,
                                                                                                                 w + 2 * f * pad,
                                                                                                                 h + 2 * f * pad)


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0):
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
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
        # batched image tensor
    else:
        b, c, w, h = x.shape
        before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f,
                                                                                              h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included.
    see our supple for more details.
    '''

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dilation, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=dilation,
                               stride=1, groups=dw_channel, dilation=dilation)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        z = y + x * self.gamma

        return z


class Downsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (hd h) (wd w) -> b (c hd wd) h w', h=self.dilation ** 2, w=self.dilation ** 2)
        x = rearrange(x, 'b c (hn hh) (wn ww) -> b c (hn wn) hh ww', hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b (c hd wd) cc hh ww-> b (c cc) (hd hh) (wd ww)', hd=H // (self.dilation ** 2),
                      wd=W // (self.dilation ** 2))
        return x


class Upsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b (c cc) (hd hh) (wd ww) -> b (c hd wd) cc hh ww', cc=self.dilation ** 2, hh=self.dilation,
                      ww=self.dilation)
        x = rearrange(x, 'b c (hn wn) hh ww -> b c (hn hh) (wn ww)', hn=self.dilation, wn=self.dilation)
        x = rearrange(x, 'b (c hd wd) h w -> b c (hd h) (wd w)', hd=H // self.dilation, wd=W // self.dilation)
        return x


class PUCA(nn.Module):
    def __init__(self, img_channel, pd, dilation, width,
                 enc_blk_nums, middle_blk_nums, dec_blk_nums):
        super().__init__()

        self.training = True
        self.pd = pd
        self.dilation = dilation

        self.intro = nn.Conv2d(img_channel, width, kernel_size=1, stride=1)
        self.tail = nn.Sequential(nn.Conv2d(width, width, kernel_size=1),
                                  nn.Conv2d(width, width // 2, kernel_size=1),
                                  nn.Conv2d(width // 2, width // 2, kernel_size=1),
                                  nn.Conv2d(width // 2, img_channel, kernel_size=1))

        self.masked_conv = nn.Sequential(
            CentralMaskedConv2d(width, width, kernel_size=2 * dilation - 1, stride=1, padding=dilation - 1),
            nn.Conv2d(width, width, kernel_size=1, stride=1),
            nn.Conv2d(width, width, kernel_size=1, stride=1))
        self.final = nn.Conv2d(width, width, kernel_size=1, stride=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dilation) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Sequential(nn.Conv2d(chan, chan // 2, kernel_size=1, stride=1),
                              Downsample(dilation)
                              )
            )

            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, dilation) for _ in range(middle_blk_nums)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1),
                    Upsample(dilation)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dilation) for _ in range(num)]

                )
            )

    def forward(self, x, refine=False, option=None):
        if option:
            self.pd[1] = option

        x = self.intro(x)
        if self.training:
            pd = self.pd[0]
        elif refine:
            pd = self.pd[2]
        else:
            pd = self.pd[1]

        b, c, h, w = x.shape
        if pd > 1:
            p = 0
            x = pixel_shuffle_down_sampling(x, pd, self.dilation)
        else:
            p = 2 * self.dilation
            x = F.pad(x, (p, p, p, p), 'reflect')

        x = self.masked_conv(x)
        encs = []
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.final(x)

        if pd > 1:
            x = pixel_shuffle_up_sampling(x, pd, self.dilation)
        if p == 0:
            x = x
        else:
            x = x[:, :, p:-p, p:-p]
        x = self.tail(x)
        return x

if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model = PUCA(            img_channel=3,
            pd = [4, 2, 1],
            dilation = 2,
            width =  128,
            enc_blk_nums = [3,4],
            middle_blk_nums = 4,
            dec_blk_nums = [4,3])

    # ptflops cal#
    flops_count, params_count = get_model_complexity_info(
        model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops_count)
    print('params: ', params_count)