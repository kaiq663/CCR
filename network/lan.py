import torch
import torch.nn as nn

class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RB(nn.Module):
    def __init__(self, filters):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 1)
        self.cuca = CALayer(channel=filters)

    def forward(self, x):
        c0 = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = self.cuca(x)
        return out + c0

class NRB(nn.Module):
    def __init__(self, n, filters):
        super(NRB, self).__init__()
        nets = []
        for i in range(n):
            nets.append(RB(filters))
        self.body = nn.Sequential(*nets)
        self.tail = nn.Conv2d(filters, filters, 1)


    def forward(self, x):
        tmp = x
        x = self.body(x)
        x = self.tail(x)
        x = x + tmp
        return x

# class LAN(nn.Module):
#     def __init__(self, blindspot, in_ch=3, out_ch=None, rbs=6):
#         super(LAN, self).__init__()
#         self.receptive_feild = blindspot
#         assert self.receptive_feild % 2 == 1
#         self.in_ch = in_ch
#         self.out_ch = self.in_ch if out_ch is None else out_ch
#         self.mid_ch = 64
#         self.rbs = rbs
#
#         layers = []
#         layers.append(nn.Conv2d(self.in_ch, self.mid_ch, 1))
#         layers.append(nn.ReLU())
#
#         for i in range(self.receptive_feild // 2):
#             layers.append(nn.Conv2d(self.mid_ch, self.mid_ch, 3, 1, 1))
#             layers.append(nn.ReLU())
#
#         layers.append(NRB(self.rbs, self.mid_ch))
#         layers.append(nn.Conv2d(self.mid_ch, self.out_ch, 1))
#
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.conv(x)

class LAN(nn.Module):
    def __init__(self, blindspot, in_ch=3, out_ch=None, enc_blk_nums=[1,1], middle_blk_nums=2, dec_blk_nums=[1,1]):
        super(LAN, self).__init__()
        self.receptive_feild = blindspot
        assert self.receptive_feild % 2 == 1
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.mid_ch = 64

        self.intro = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, 1),
            nn.ReLU()
        )

        layers = []
        for i in range(self.receptive_feild // 2):
            layers.append(nn.Conv2d(self.mid_ch, self.mid_ch, 3, 1, 1))
            layers.append(nn.ReLU())
        self.rfconv = nn.Sequential(*layers)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    NRB(n=num, filters=self.mid_ch)
                )
            )
            self.downs.append(
                nn.Sequential(nn.Conv2d(self.mid_ch, self.mid_ch * 2, kernel_size=1, stride=1),
                              nn.ReLU(),
                              nn.MaxPool2d(2)
                              )
                )
            self.mid_ch = self.mid_ch * 2

        self.middle_blks = nn.Sequential(
                NRB(n=middle_blk_nums, filters=self.mid_ch)
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(self.mid_ch, self.mid_ch // 2, 1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                )
            )
            self.mid_ch = self.mid_ch // 2
            self.decoders.append(
                nn.Sequential(
                    NRB(n=num, filters=self.mid_ch)
                )
            )

        self.final = nn.Sequential(
            nn.Conv2d(self.mid_ch, self.mid_ch, 1),
            nn.ReLU()
        )

        self.tail = nn.Conv2d(self.mid_ch, self.out_ch, 1)

    def forward(self, x):

        x = self.intro(x)
        x = self.rfconv(x)

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

        x = self.tail(x)
        return x

