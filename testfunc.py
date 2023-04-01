import models
import torch
import functions
import numpy
import os
from skimage import io
import argparse
import config

def test_matRead(data):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/3000.
    data=torch.from_numpy(data)
    data = data.to(torch.device('cuda')).type(torch.cuda.FloatTensor)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mspath', help='test lrms image name', default='F:/dwx/datasets/test256/Birdtest/SimulatedRawMS/')
    parser.add_argument('--panpath', help='test hrpan image name', default='F:/dwx/datasets/test256/Birdtest/SimulatedRawPAN/')
    parser.add_argument('--modelpath', help='output model dir', default='./checkpoints/best.pth')
    parser.add_argument('--saveimgpath', help='output model dir', default='./output/Bird/sim/')
    parser.add_argument('--device', default=torch.device('cuda'))
    opt = parser.parse_args()
    c = config.configuration()
    net = models.Generator_A2B(c)
    net = torch.nn.DataParallel(net).cuda()
    modelname = opt.modelpath
    net.load_state_dict(torch.load(modelname))
    for msfilename in os.listdir(opt.mspath):
        num = msfilename.split('m')[0]
        print(opt.mspath + msfilename)
        ms_val = io.imread(opt.mspath + msfilename)
        ms_val = test_matRead(ms_val)
        ms_val = torch.nn.functional.interpolate(ms_val, size=(256, 256), mode='bilinear')
        panname = msfilename.split('m')[0] + 'p.tif'
        pan_val = io.imread(opt.panpath + panname)
        pan_val = pan_val[:, :, None]
        pan_val = test_matRead(pan_val)
        in_s = net(pan_val, ms_val)
        outname = opt.saveimgpath + num + '.tif'
        print(outname)
        io.imsave(outname, functions.convert_image_np(in_s.detach(), opt).astype(numpy.uint16))

if __name__ == '__main__':
    main()