import torch
import numpy as np
from sewar.full_ref import sam
import torch.nn as nn

def matRead(data,opt):
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    return data

def test_matRead(data,opt):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    return data

def getBatch(ms_data,pan_data,gt_data, bs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def getTest(ms_data,pan_data,gt_data,bs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def convert_image_np(inp,opt):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    inp=inp*3000.
    return inp

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img,hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.detach().numpy()
    sr_img=sr_img[-1,:,:,:]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.detach().numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value

def SAMLoss(sr_img,hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.detach().numpy()
    sr_img=sr_img[-1,:,:,:]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.detach().numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    sam_value = torch.tensor(sam_value).cuda()
    sam_value.requires_grad_(True)
    return sam_value

def gradientLoss_MS(middle_image):
    channelsGradient_x=np.zeros([middle_image.shape[2],middle_image.shape[3]])
    channelsGradient_x=(torch.tensor(channelsGradient_x, requires_grad=True)).to(torch.device('cuda')).type(torch.cuda.FloatTensor)
    channelsGradient_y=np.zeros([middle_image.shape[2],middle_image.shape[3]])
    channelsGradient_y=(torch.tensor(channelsGradient_y, requires_grad=True)).to(torch.device('cuda')).type(torch.cuda.FloatTensor)
    for i in range(4):
        x,y=gradient(middle_image[:,i,:,:])
        channelsGradient_x=0.25*x+channelsGradient_x
        channelsGradient_y=0.25*y+channelsGradient_y
    return channelsGradient_x[None,None,:,:],channelsGradient_y[None,None,:,:]

def gradientLoss_P(pan_image):
    pan_image = pan_image[-1, :, :, :]
    grayGradient_x,grayGradient_y=gradient(pan_image)
    return grayGradient_x[None, None, :, :], grayGradient_y[None, None, :, :]

def gradient(image):
     image = image.to(torch.device('cpu'))
     image = np.array(image.detach())
     dx, dy = np.gradient(image[0,:,:], edge_order=1)
     dx = (torch.tensor(dx, requires_grad=True)).to(torch.device('cuda')).type(torch.cuda.FloatTensor)
     dy = (torch.tensor(dy, requires_grad=True)).to(torch.device('cuda')).type(torch.cuda.FloatTensor)
     return dx, dy

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
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        out = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return out