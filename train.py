import copy
import os
import pdb

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch import autograd
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import config
import functions
import models
import dataloader
import train_DegradationMatrix

def GP_Afuc(D, real, fake, config, device):
    alpha = torch.rand(real.shape[0], 1, 1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.float().to(device)
    xhat = alpha * real + (1 - alpha) * fake
    xhat = xhat.float().to(device)
    xhat = autograd.Variable(xhat, requires_grad=True)
    xhat_D = D(xhat)
    grad = autograd.grad(outputs=xhat_D, inputs=xhat, grad_outputs=torch.ones(xhat_D.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * config.eta
    return penalty

def GP_Afuc_ms(D, real, fake, config, device):
    alpha = torch.rand(real.shape[0], 4, 1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.float().to(device)
    xhat = alpha * real + (1 - alpha) * fake
    xhat = xhat.float().to(device)
    xhat = autograd.Variable(xhat, requires_grad=True)
    xhat_D = D(xhat)
    grad = autograd.grad(outputs=xhat_D, inputs=xhat, grad_outputs=torch.ones(xhat_D.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * config.eta
    return penalty

def train(config, device, set, test_set):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    GA2B = models.Generator_A2B(config)
    DA = models.Discriminator_A(config)
    GB2A_ms = models.Generator_B2A_ms(channels=4)
    DA_ms = models.Discriminator_A_ms(config)
    GB2A = train_DegradationMatrix.Generator_B2A_pan_involution(config)

    GA2B = nn.DataParallel(GA2B).cuda()
    DA = nn.DataParallel(DA).cuda()
    GB2A = nn.DataParallel(GB2A).cuda()
    GB2A.load_state_dict(torch.load('./checkpoints/modelB2A_pan_Bird_involution_size64_2/best.pth'))
    GB2A_ms = nn.DataParallel(GB2A_ms).cuda()
    DA_ms = nn.DataParallel(DA_ms).cuda()

    best_weight = copy.deepcopy(GA2B.state_dict())
    best_epoch = 0
    best_SAM = 1.0
    loss = torch.nn.L1Loss()
    loss_mse = torch.nn.MSELoss()

    print("Training start ...")

    for e in range(config.epochs):

        print("Epoch {} started ...".format(e))
        print("Training Discriminator ... Intermediate result can be found in "+config.result_dir)
        
        G_lr = config.G_lr
        D_lr = config.D_lr
        
        if config.lr_decay=='linear' and config.decay_epoch<(e+1):
            print("Learning rate decay ... Option : {}".format(config.lr_decay))
            G_lr = config.G_lr * ((e+1)/config.epochs)
            D_lr = config.D_lr * ((e+1)/config.epochs)

        optimizer_GA2B = optim.Adam(GA2B.parameters(), lr=G_lr, betas=(config.beta1, config.beta2))
        optimizer_DA = optim.Adam(itertools.chain(DA.parameters(), DA_ms.parameters()), lr=D_lr, betas=(config.beta1, config.beta2))

        DA.train()
        DA_ms.train()

        for batch_idx, (pan, ms, gt, ms_ori) in enumerate(set):
            if torch.cuda.is_available():
                batch_C, batch_A, batch_B, batch_A_ori = pan.cuda(), ms.cuda(), gt.cuda(), ms_ori.cuda()
                batch_A = Variable(batch_A.to(torch.float32))
                batch_B = Variable(batch_B.to(torch.float32))
                batch_C = Variable(batch_C.to(torch.float32))
                batch_A_ori = Variable(batch_A_ori.to(torch.float32))

            optimizer_DA.zero_grad()  # MS判别器

            D_real_loss_A_ms = DA_ms(batch_A_ori).mean()
            fake_A2B = GA2B(batch_C, batch_A)
            fake_B2A_ms = GB2A_ms(fake_A2B)
            D_fake_loss_A_ms = DA_ms(fake_B2A_ms).mean()
            GP_A_ms = GP_Afuc_ms(DA_ms, batch_A_ori, fake_B2A_ms, config, device)
            loss_OTDisc_A_ms = -D_real_loss_A_ms + D_fake_loss_A_ms + GP_A_ms
            loss_OTDisc_A_ms.backward(retain_graph=True)

            D_real_loss_A = DA(batch_C).mean()
            fake_B2A = GB2A(fake_A2B)
            D_fake_loss_A = DA(fake_B2A).mean()
            GP_A = GP_Afuc(DA, batch_C, fake_B2A, config, device)
            loss_OTDisc_A = -D_real_loss_A + D_fake_loss_A + GP_A
            loss_OTDisc_A.backward(retain_graph=True)
            optimizer_DA.step()

            if (batch_idx+1) % config.print_step == 0:
                training_state = '  '.join(
                    ['epoch:{}', '[{} / {}]', '-D_real_loss_A: {:.6f}', 'D_fake_loss_A: {:.6f}', 'GP_A: {:.6f}',
                     '-D_real_loss_A_ms: {:.6f}', 'D_fake_loss_A_ms: {:.6f}', 'GP_A_ms: {:.6f}',
                     ]
                )
                training_state = training_state.format(
                    e, batch_idx, len(set), -D_real_loss_A, D_fake_loss_A, GP_A,
                    -D_real_loss_A_ms, D_fake_loss_A_ms, GP_A_ms,
                )
                print(training_state)

        print("Training Generator ... Intermediate result can be found in "+config.result_dir)
        guass = functions.Generator_B2A_ms(1).to(torch.device('cuda'))
        GA2B.train()

        for batch_idx, (pan, ms, gt, ms_ori) in enumerate(set):
            if torch.cuda.is_available():
                batch_C, batch_A, batch_B, batch_A_ori = pan.cuda(), ms.cuda(), gt.cuda(), ms_ori.cuda()
                batch_A = Variable(batch_A.to(torch.float32))
                batch_B = Variable(batch_B.to(torch.float32))
                batch_C = Variable(batch_C.to(torch.float32))
                batch_A_ori = Variable(batch_A_ori.to(torch.float32))

            optimizer_GA2B.zero_grad()
            fake_A2B = GA2B(batch_C, batch_A)
            fake_B2A_ms = GB2A_ms(fake_A2B)
            loss_OTDIcs_B_ms = -DA_ms(fake_B2A_ms).mean()
            loss_OTDIcs_B_ms.backward(retain_graph=True)
            fake_B2A = GB2A(fake_A2B)
            loss_OTDIcs_B = -DA(fake_B2A).mean()
            loss_OTDIcs_B.backward(retain_graph=True)

            pan_loss = 50. * loss(batch_C, fake_B2A)
            ms_loss = 30. * loss(batch_A_ori, fake_B2A_ms)
            SAMloss = 100. * functions.SAMLoss(fake_B2A_ms, batch_A_ori)
            middle_image_gradient_x, middle_image_gradient_y = functions.gradientLoss_MS(fake_A2B)
            pan_image_gradient_x, pan_image_gradient_y = functions.gradientLoss_P(batch_C)
            gradient_loss_x = loss_mse(middle_image_gradient_x, pan_image_gradient_x)
            gradient_loss_y = loss_mse(middle_image_gradient_y, pan_image_gradient_y)
            gradient_loss = 100. * (gradient_loss_x + gradient_loss_y)
            g_loss = ms_loss+pan_loss+SAMloss+gradient_loss
            g_loss.backward(retain_graph=True)
            optimizer_GA2B.step()

            if (batch_idx+1) % config.print_step == 0:
                training_state = '  '.join(
                    [
                        'epoch:{}', '[{} / {}]',
                        'gradient_loss:{:.6f}', 'SAMloss:{:.6f}',  'panloss:{:.6f}', 'msloss:{:.6f}']
                )
                training_state = training_state.format(
                    e, batch_idx, len(set),
                    gradient_loss, SAMloss, pan_loss, ms_loss
                )
                print(training_state)
        
        print("Saving Generator at {}".format(config.checkpoint_dir) + " ...")
        
        torch.save(GA2B.state_dict(), os.path.join(config.checkpoint_dir, 'modelA2B_{:04d}'.format(e+1) + '.pth'))

        GA2B.eval()
        epoch_SAM = functions.AverageMeter()
        for batch_idx, (pan, ms, gt, ms_ori) in enumerate(test_set):
            if torch.cuda.is_available():
                batch_C, batch_A, batch_B, batch_A_ori = pan.cuda(), ms.cuda(), gt.cuda(), ms_ori.cuda()
                batch_A = Variable(batch_A.to(torch.float32))
                batch_B = Variable(batch_B.to(torch.float32))
                batch_C = Variable(batch_C.to(torch.float32))
                batch_A_ori = Variable(batch_A_ori.to(torch.float32))
            fake_A2B = GA2B(batch_C, batch_A)
            test_SAM = functions.SAM(fake_A2B, batch_B)
            if test_SAM == test_SAM:
                epoch_SAM.update(test_SAM, batch_A.shape[0])
        print('eval SAM: {:.6f}, epoch_SAM: {:.6f}'.format(epoch_SAM.avg, best_SAM))

        if epoch_SAM.avg < best_SAM:
            best_epoch = e
            best_SAM = epoch_SAM.avg
            best_weight = copy.deepcopy(GA2B.state_dict())
            print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
            torch.save(best_weight, os.path.join(config.checkpoint_dir,  'best.pth'))
            
def main():
    
    c = config.configuration()
    print(c)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    train_set = dataloader.get_training_set('F:/dwx/datasets/train256/train256/Birdtrain/')
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=c.batch_size, shuffle=True)
    test_set = dataloader.get_training_set('F:/dwx/datasets/val256/Birdval1/')
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=c.batch_size, shuffle=True)
    print("Number of train data: {}".format(len(train_loader)))
    print("Number of test data: {}".format(len(test_loader)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(c, device, train_loader, test_loader)
    print("Exiting ...")
    
if __name__ == '__main__':
    main()
