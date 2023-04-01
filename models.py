import pdb

import torch
from torch import nn
import numpy

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('pad',nn.ReflectionPad2d(1)),
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        # self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

class Generator_B2A(nn.Module):
    def __init__(self,config):
        super(Generator_B2A, self).__init__()
        dataR = [[0.25]]  # 2308
        dataR = torch.FloatTensor(dataR).unsqueeze(0)
        dataG = [[0.25]]  # 2315
        dataG = torch.FloatTensor(dataG).unsqueeze(0)
        dataB = [[0.25]]  # 1139
        dataB = torch.FloatTensor(dataB).unsqueeze(0)
        dataN = [[0.25]]  # 4239
        dataN = torch.FloatTensor(dataN).unsqueeze(0)
        self.weight_R = nn.Parameter(data=dataR, requires_grad=False)
        self.weight_G = nn.Parameter(data=dataG, requires_grad=False)
        self.weight_B = nn.Parameter(data=dataB, requires_grad=False)
        self.weight_N = nn.Parameter(data=dataN, requires_grad=False)

    def forward(self, x):
        h5 = x[:, 0, :, :]*self.weight_R + x[:, 1, :, :]*self.weight_G+x[:, 2, :, :]*self.weight_B+x[:, 3, :, :]*self.weight_N
        return h5[:, None, :, :]

class Generator_B2A_ms(nn.Module):
    def __init__(self, channels=4):
        super(Generator_B2A_ms, self).__init__()
        self.channels = channels
        kernel = [[0.0265, 0.0354, 0.0390, 0.0354, 0.0265],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0390, 0.0520, 0.0573, 0.0520, 0.0390],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0265, 0.0354, 0.0390, 0.0354, 0.0265]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = numpy.repeat(kernel, self.channels, axis=0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        # pdb.set_trace()
        out = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        out = torch.nn.functional.interpolate(out, size=(64, 64), mode='bilinear', align_corners=True)
        return out

class Generator_A2B(nn.Module):
    def __init__(self, config):
        super(Generator_A2B, self).__init__()
        self.layer1 = ConvBlock(in_channel=5, out_channel=config.n_feature, ker_size=3, padd=0, stride=1)
        self.layer2 = ConvBlock(in_channel=5+config.n_feature, out_channel=config.n_feature, ker_size=3, padd=0, stride=1)
        self.layer3 = ConvBlock(in_channel=5+config.n_feature*2, out_channel=config.n_feature, ker_size=3, padd=0, stride=1)
        self.layer4 = ConvBlock(in_channel=5+config.n_feature*3, out_channel=config.n_feature, ker_size=3, padd=0, stride=1)
        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=5+config.n_feature*4, out_channels=4, kernel_size=3, padding=0)
        )

    def forward(self, pan, ms):
        x = torch.cat([pan, ms], dim=1)
        x1 = self.layer1(x)
        x2 = self.layer2(torch.cat([x, x1], dim=1))
        x3 = self.layer3(torch.cat([x, x1, x2], dim=1))
        x4 = self.layer4(torch.cat([x, x1, x2, x3], dim=1))
        x5 = self.layer5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5


class Discriminator_A(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channel = config.channel
        self.n_feature = config.n_feature
        self.kernel_size = config.D_kernel_size
        self.stride = config.D_stride
        self.factor = config.factor
        
        self.conv1 = nn.Conv2d(1, self.n_feature, self.kernel_size, stride=self.stride)
        self.leaky1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature * self.factor, self.kernel_size, stride=self.stride)
        # self.instance2 = nn.InstanceNorm2d(self.n_feature * self.factor)
        self.leaky2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(self.n_feature * self.factor, self.n_feature * (self.factor**2), self.kernel_size, stride=self.stride)
        # self.instance3 = nn.InstanceNorm2d(self.n_feature * (self.factor**2))
        self.leaky3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv2d(self.n_feature * (self.factor**2), self.n_feature * (self.factor**3), self.kernel_size, stride=self.stride//self.factor, padding=1)
        # self.instance4 = nn.InstanceNorm2d(self.n_feature * (self.factor**3))
        self.leaky4 = nn.LeakyReLU()
        
        self.last = nn.Conv2d(self.n_feature * (self.factor**3), 1, self.kernel_size, stride=self.stride//self.factor, padding=1)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.leaky1(x)
        
        x = self.conv2(x)
        # x = self.instance2(x)
        x = self.leaky2(x)
        
        x = self.conv3(x)
        # x = self.instance3(x)
        x = self.leaky3(x)
        
        x = self.conv4(x)
        # x = self.instance4(x)
        x = self.leaky4(x)
        
        x = self.last(x)
        x = self.output(x)
        
        return x


class Discriminator_A_ms(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channel = config.channel
        self.n_feature = config.n_feature
        self.kernel_size = config.D_kernel_size
        self.stride = config.D_stride
        self.factor = config.factor

        self.conv1 = nn.Conv2d(4, self.n_feature, self.kernel_size, stride=self.stride)
        self.leaky1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature * self.factor, self.kernel_size, stride=self.stride)
        # self.instance2 = nn.InstanceNorm2d(self.n_feature * self.factor)
        self.leaky2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(self.n_feature * self.factor, self.n_feature * (self.factor ** 2), self.kernel_size,
                               stride=self.stride)
        # self.instance3 = nn.InstanceNorm2d(self.n_feature * (self.factor**2))
        self.leaky3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(self.n_feature * (self.factor ** 2), self.n_feature * (self.factor ** 3),
                               self.kernel_size, stride=self.stride // self.factor, padding=1)
        # self.instance4 = nn.InstanceNorm2d(self.n_feature * (self.factor**3))
        self.leaky4 = nn.LeakyReLU()

        self.last = nn.Conv2d(self.n_feature * (self.factor ** 3), 4, self.kernel_size,
                              stride=self.stride // self.factor, padding=1)
        # self.last = nn.Conv2d(self.n_feature * (self.factor ** 3), 4, self.kernel_size,
        #                       stride=self.stride, padding=1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.leaky1(x)

        x = self.conv2(x)
        # x = self.instance2(x)
        x = self.leaky2(x)

        x = self.conv3(x)
        # x = self.instance3(x)
        x = self.leaky3(x)

        x = self.conv4(x)
        # x = self.instance4(x)
        x = self.leaky4(x)

        x = self.last(x)
        x = self.output(x)

        return x