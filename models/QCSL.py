import torch
import torch.nn as nn

from torch.nn.init import kaiming_normal_, constant_
from models.quaternion_layer import *
from models.non_local_dot_product import NONLocalBlock3D

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            QuaternionConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.1),

            nn.Dropout(p=0.2)
        )
    else:
        return nn.Sequential(
            # in:NxCxDxHxW out:NxCoutxDoutxHoutxWout
            QuaternionConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        QuaternionTransposeConv(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class fusion_self_attention(nn.Module):
    def __init__(self, in_channel):
        super(fusion_self_attention, self).__init__()
        self.in_channel = in_channel
        self.convQ = conv(batchNorm=True, in_planes=self.in_channel, out_planes=self.in_channel,
                          kernel_size=3, stride=1)
        self.convK = conv(batchNorm=True, in_planes=self.in_channel, out_planes=self.in_channel,
                          kernel_size=3, stride=1)
        self.convV = conv(batchNorm=True, in_planes=self.in_channel, out_planes=self.in_channel,
                          kernel_size=3, stride=1)
        self.hamilton_product = hamilton_product
        self.weight = nn.Parameter(torch.zeros(4), requires_grad=True)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, a, b, c, d):
        aQ = self.convQ(a)
        bQ = self.convQ(b)
        cQ = self.convQ(c)
        dQ = self.convQ(d)

        aK = self.convK(a)
        bK = self.convK(b)
        cK = self.convK(c)
        dK = self.convK(d)

        aV = self.convK(a)
        bV = self.convK(b)
        cV = self.convK(c)
        dV = self.convK(d)

        aKQ = self.hamilton_product(aQ, aK).unsqueeze(2)
        bKQ = self.hamilton_product(bQ, bK).unsqueeze(2)
        cKQ = self.hamilton_product(cQ, cK).unsqueeze(2)
        dKQ = self.hamilton_product(dQ, dK).unsqueeze(2)

        aKQ = self.softmax(aKQ).squeeze(2)
        bKQ = self.softmax(bKQ).squeeze(2)
        cKQ = self.softmax(cKQ).squeeze(2)
        dKQ = self.softmax(dKQ).squeeze(2)

        outa = self.hamilton_product(aKQ, aV)
        outb = self.hamilton_product(bKQ, bV)
        outc = self.hamilton_product(cKQ, cV)
        outd = self.hamilton_product(dKQ, dV)

        out = self.weight[0]*outa + self.weight[1]*outb + self.weight[2]*outc + self.weight[3]*outd

        return out+a, out+b, out+c, out+d


class QCSL(nn.Module):
    def __init__(self, in_channels=4, classes=4, batchNorm=True, args=None):
        super(QCSL, self).__init__()
        assert in_channels == 4, "error:need 4 channels"

        self.batchNorm = batchNorm
        self.assign_ch = 4  # 6/18/26
        # s=1, not change dimension
        self.conva4 = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)  
        self.convb4 = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.convc4 = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.convd4 = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        # s=1, not change dimension, learn the coordinate
        self.conv_a1t = conv(self.batchNorm, 4, 16, kernel_size=3, stride=1)  
        self.conv_b1t = conv(self.batchNorm, 4, 16, kernel_size=3, stride=1)
        self.conv_c1t = conv(self.batchNorm, 4, 16, kernel_size=3, stride=1)
        self.conv_d1t = conv(self.batchNorm, 4, 16, kernel_size=3, stride=1)

        self.conv_a1 = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)  # s=2, half dimension，p=1
        self.conv_b1 = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv_c1 = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv_d1 = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        # 2
        self.conv_a2 = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv_b2 = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv_c2 = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv_d2 = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        # 3
        self.conv_a3 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv_b3 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv_c3 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv_d3 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        # 4
        self.conv_a4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv_b4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv_c4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv_d4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)

        # d4
        self.deconv_a4 = deconv(256, 128)
        self.dc_a4 = conv(self.batchNorm, 256, 128)
        self.deconv_b4 = deconv(256, 128)
        self.dc_b4 = conv(self.batchNorm, 256, 128)
        self.deconv_c4 = deconv(256, 128)
        self.dc_c4 = conv(self.batchNorm, 256, 128)
        self.deconv_d4 = deconv(256, 128)
        self.dc_d4 = conv(self.batchNorm, 256, 128)

        self.att4 = fusion_self_attention(128)

        # d3
        self.deconv_a3 = deconv(128, 64)
        self.dc_a3 = conv(self.batchNorm, 128, 64)
        self.deconv_b3 = deconv(128, 64)
        self.dc_b3 = conv(self.batchNorm, 128, 64)
        self.deconv_c3 = deconv(128, 64)
        self.dc_c3 = conv(self.batchNorm, 128, 64)
        self.deconv_d3 = deconv(128, 64)
        self.dc_d3 = conv(self.batchNorm, 128, 64)

        self.att3 = fusion_self_attention(64)

        # d2
        self.deconv_a2 = deconv(64, 32)
        self.dc_a2 = conv(self.batchNorm, 64, 32)
        self.deconv_b2 = deconv(64, 32)
        self.dc_b2 = conv(self.batchNorm, 64, 32)
        self.deconv_c2 = deconv(64, 32)
        self.dc_c2 = conv(self.batchNorm, 64, 32)
        self.deconv_d2 = deconv(64, 32)
        self.dc_d2 = conv(self.batchNorm, 64, 32)

        self.att2 = fusion_self_attention(32)

        # d1
        self.deconv_a1 = deconv(32, 16)
        self.dc_a1 = conv(self.batchNorm, 32, 8)
        self.deconv_b1 = deconv(32, 16)
        self.dc_b1 = conv(self.batchNorm, 32, 8)
        self.deconv_c1 = deconv(32, 16)
        self.dc_c1 = conv(self.batchNorm, 32, 8)
        self.deconv_d1 = deconv(32, 16)
        self.dc_d1 = conv(self.batchNorm, 32, 8)

        # self.att1 = fusion_self_attention(8) # please use it under enough cuda memory 

        self.assc = nn.Conv3d(32, 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.softmax = nn.Softmax(1)

    def forward(self, inputs):
        ya_in = self.conva4(inputs[:, 0:1, ...]) 
        yb_in = self.convb4(inputs[:, 1:2, ...])
        yc_in = self.convc4(inputs[:, 2:3, ...])
        yd_in = self.convd4(inputs[:, 3:4, ...])

        ya_1t = self.conv_a1t(ya_in)
        yb_1t = self.conv_b1t(yb_in)
        yc_1t = self.conv_c1t(yc_in)
        yd_1t = self.conv_d1t(yd_in)

        ya_1 = self.conv_a1(ya_1t)
        yb_1 = self.conv_b1(yb_1t)
        yc_1 = self.conv_c1(yc_1t)
        yd_1 = self.conv_d1(yd_1t)

        ya_2 = self.conv_a2(ya_1)
        yb_2 = self.conv_b2(yb_1)
        yc_2 = self.conv_c2(yc_1)
        yd_2 = self.conv_d2(yd_1)

        ya_3 = self.conv_a3(ya_2)
        yb_3 = self.conv_b3(yb_2)
        yc_3 = self.conv_c3(yc_2)
        yd_3 = self.conv_d3(yd_2)

        ya_4 = self.conv_a4(ya_3)
        yb_4 = self.conv_b4(yb_3)
        yc_4 = self.conv_c4(yc_3)
        yd_4 = self.conv_d4(yd_3)

        # d4
        de_a = self.deconv_a4(ya_4)
        de_a = torch.cat((de_a, ya_3), dim=1)
        de_a = self.dc_a4(de_a)

        de_b = self.deconv_b4(yb_4)
        de_b = torch.cat((de_b, yb_3), dim=1)
        de_b = self.dc_b4(de_b)

        de_c = self.deconv_c4(yc_4)
        de_c = torch.cat((de_c, yc_3), dim=1)
        de_c = self.dc_c4(de_c)

        de_d = self.deconv_d4(yd_4)
        de_d = torch.cat((de_d, yd_3), dim=1)
        de_d = self.dc_d4(de_d)
        # attention
        de_a , de_b, de_c, de_d = self.att4(de_a, de_b, de_c, de_d)

        # d3
        de_a = self.deconv_a3(de_a)
        de_a = torch.cat((de_a, ya_2), dim=1)
        de_a = self.dc_a3(de_a)

        de_b = self.deconv_b3(de_b)
        de_b = torch.cat((de_b, yb_2), dim=1)
        de_b = self.dc_b3(de_b)

        de_c = self.deconv_c3(de_c)
        de_c = torch.cat((de_c, yc_2), dim=1)
        de_c = self.dc_c3(de_c)

        de_d = self.deconv_d3(de_d)
        de_d = torch.cat((de_d, yd_2), dim=1)
        de_d = self.dc_d3(de_d)

        de_a , de_b, de_c, de_d = self.att3(de_a, de_b, de_c, de_d)
        # d2
        de_a = self.deconv_a2(de_a)
        de_a = torch.cat((de_a, ya_1), dim=1)
        de_a = self.dc_a2(de_a)

        de_b = self.deconv_b2(de_b)
        de_b = torch.cat((de_b, yb_1), dim=1)
        de_b = self.dc_b2(de_b)

        de_c = self.deconv_c2(de_c)
        de_c = torch.cat((de_c, yc_1), dim=1)
        de_c = self.dc_c2(de_c)

        de_d = self.deconv_d2(de_d)
        de_d = torch.cat((de_d, yd_1), dim=1)
        de_d = self.dc_d2(de_d)

        de_a, de_b, de_c, de_d = self.att2(de_a, de_b, de_c, de_d)

        # d1
        de_a = self.deconv_a1(de_a)
        de_a = torch.cat((de_a, ya_1t), dim=1)
        de_a = self.dc_a1(de_a)

        de_b = self.deconv_b1(de_b)
        de_b = torch.cat((de_b, yb_1t), dim=1)
        de_b = self.dc_b1(de_b)

        de_c = self.deconv_c1(de_c)
        de_c = torch.cat((de_c, yc_1t), dim=1)
        de_c = self.dc_c1(de_c)

        de_d = self.deconv_d1(de_d)
        de_d = torch.cat((de_d, yd_1t), dim=1)
        de_d = self.dc_d1(de_d)

        # de_a, de_b, de_c, de_d = self.att1(de_a, de_b, de_c, de_d)

        input_o = torch.cat((de_a, de_b, de_c, de_d), dim=1)
        mask = self.assc(input_o)
        prob0 = self.softmax(mask)
        return prob0


