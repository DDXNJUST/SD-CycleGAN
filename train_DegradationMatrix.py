import copy
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import config
import functions
import dataloader
from mmcv.cnn import ConvModule

class Generator_B2A_pan_involution(nn.Module):
    def __init__(self, config):
        super(Generator_B2A_pan_involution, self).__init__()
        reduction_ratio=4
        self.group_channels=4
        self.groups=8
        self.conv1=ConvModule(
            in_channels=32,
            out_channels=32//reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        self.conv2 = ConvModule(
            in_channels=32 // reduction_ratio,
            out_channels=1**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        self.unflod = nn.Unfold(1,1,0,1)
        self.inc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4,32,kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 1, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        in_x = self.inc(x)
        weight = self.conv2(self.conv1(in_x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, 1 ** 2, h, w).unsqueeze(2)
        out = self.unflod(in_x).view(b, self.groups, self.group_channels, 1 ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, 32, h, w)
        out_x = self.outc(out + in_x)
        return out_x

def train(config, device, set, test_set):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    GB2A_pan = Generator_B2A_pan_involution(config)
    GB2A_pan = nn.DataParallel(GB2A_pan).cuda()

    best_SAM = 1.0
    loss = torch.nn.L1Loss()

    print("Training start ...")

    for e in range(config.epochs):

        print("Epoch {} started ...".format(e))
        print("Training Discriminator ... Intermediate result can be found in " + config.result_dir)

        G_lr = config.G_lr

        if config.lr_decay == 'linear' and config.decay_epoch < (e + 1):
            print("Learning rate decay ... Option : {}".format(config.lr_decay))
            G_lr = config.G_lr * ((e + 1) / config.epochs)

        # optimizer_GB2A_ms = optim.Adam(GB2A_ms.parameters(), lr=G_lr, betas=(config.beta1, config.beta2))
        optimizer_GB2A_pan = optim.Adam(GB2A_pan.parameters(), lr=G_lr, betas=(config.beta1, config.beta2))

        GB2A_pan.train()

        for batch_idx, (pan, ms, gt, ms_ori) in enumerate(set):
            if torch.cuda.is_available():
                batch_C, batch_A, batch_B, batch_A_ori = pan.cuda(), ms.cuda(), gt.cuda(), ms_ori.cuda()
                batch_A = Variable(batch_A.to(torch.float32))
                batch_B = Variable(batch_B.to(torch.float32))
                batch_C = Variable(batch_C.to(torch.float32))
                batch_A_ori = Variable(batch_A_ori.to(torch.float32))

            optimizer_GB2A_pan.zero_grad()
            # pdb.set_trace()
            fake_B2A_pan = GB2A_pan(batch_A_ori)
            batch_C_64 = torch.nn.functional.interpolate(batch_C, size=(64, 64), mode='bilinear', align_corners=True)
            pan_loss = loss(fake_B2A_pan, batch_C_64)
            pan_loss.backward(retain_graph=True)
            optimizer_GB2A_pan.step()

            if (batch_idx + 1) % config.print_step == 0:
                training_state = '  '.join(
                    [
                        'epoch:{}', '[{} / {}]', 'pan_loss:{:.6f}']
                )
                training_state = training_state.format(
                    e, batch_idx, len(set), pan_loss
                )
                print(training_state)

        print("Saving Generator at {}".format(config.checkpoint_dir) + " ...")

        torch.save(GB2A_pan.state_dict(), os.path.join('F:/dwx/OT-CycleGAN/checkpoints/modelB2A_pan_Geo2/modelB2A_pan_{:04d}'.format(e + 1) + '.pth'))


        GB2A_pan.eval()
        epoch_SAM = functions.AverageMeter()
        for batch_idx, (pan, ms, gt, ms_ori) in enumerate(test_set):
            if torch.cuda.is_available():
                batch_C, batch_A, batch_B, batch_A_ori = pan.cuda(), ms.cuda(), gt.cuda(), ms_ori.cuda()
                batch_A = Variable(batch_A.to(torch.float32))
                batch_B = Variable(batch_B.to(torch.float32))
                batch_C = Variable(batch_C.to(torch.float32))
                batch_A_ori = Variable(batch_A_ori.to(torch.float32))
            fake_B2A_pan = GB2A_pan(batch_B)

            test_SAM = functions.SAM(fake_B2A_pan, batch_C)
            if test_SAM == test_SAM:
                epoch_SAM.update(test_SAM, batch_A.shape[0])
        print('eval SAM: {:.6f}, epoch_SAM: {:.6f}'.format(epoch_SAM.avg, best_SAM))

        if epoch_SAM.avg < best_SAM:
            best_epoch = e
            best_SAM = epoch_SAM.avg
            best_weight = copy.deepcopy(GB2A_pan.state_dict())
            print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
            # torch.save(best_weight, os.path.join(config.checkpoint_dir, 'best.pth'))
            torch.save(best_weight, os.path.join('F:/dwx/OT-CycleGAN/checkpoints/modelB2A_pan_Geo2/best.pth'))

def main():
    c = config.configuration()
    print(c)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    train_set = dataloader.get_training_set('F:/dwx/datasets/train256/train256/Geotrain/')
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=c.batch_size, shuffle=True)
    test_set = dataloader.get_training_set('F:/dwx/datasets/val256/Geoval1/')
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=c.batch_size, shuffle=True)
    print("Number of train data: {}".format(len(train_loader)))
    print("Number of test data: {}".format(len(test_loader)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(c, device, train_loader, test_loader)
    print("Exiting ...")


if __name__ == '__main__':
    main()