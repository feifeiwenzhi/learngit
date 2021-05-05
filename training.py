import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools

import ConSinGAN.functions as functions
import ConSinGAN.models as models


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    real, real2 = functions.read_two_domains(opt)
    # real = functions.read_image(opt)
    # print(0, real.shape)
    real = functions.adjust_scales2image(real, opt)
    reals = functions.create_reals_pyramid(real, opt)

    real2 = functions.adjust_scales2image(real2, opt)
    reals2 = functions.create_reals_pyramid(real2, opt)

    generator, generator2 = init_G(opt)
    fixed_noise = []
    noise_amp = []
    fixed_noise2 = []
    noise_amp2 = []
    for scale_num in range(opt.stop_scale+1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                print(OSError)
                pass
        functions.save_image('{}/real_scale.jpg'.format(opt.outf), reals[scale_num])

        d_curr, d_curr2 = init_D(opt)
        if scale_num > 0:
            d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))
            generator.init_next_stage()
            d_curr2.load_state_dict(torch.load('%s/%d/netD2.pth' % (opt.out_,scale_num-1)))
            generator2.init_next_stage()

        writer = SummaryWriter(log_dir=opt.outf)
        fixed_noise, noise_amp, generator, d_curr, fixed_noise2, noise_amp2, generator2, d_curr2 = \
            train_single_scale(d_curr, generator, reals, fixed_noise, noise_amp, d_curr2, generator2, reals2,
                               fixed_noise2, noise_amp2, opt, scale_num, writer)

        torch.save(fixed_noise, '%s/fixed_noise.pth' % (opt.out_))
        torch.save(generator, '%s/G.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(noise_amp, '%s/noise_amp.pth' % (opt.out_))
        torch.save(fixed_noise2, '%s/fixed_noise2.pth' % (opt.out_))
        torch.save(generator2, '%s/G2.pth' % (opt.out_))
        torch.save(reals2, '%s/reals2.pth' % (opt.out_))
        torch.save(noise_amp2, '%s/noise_amp2.pth' % (opt.out_))
        del d_curr, d_curr2
    writer.close()
    return


def train_single_scale(netD, netG, reals, fixed_noise, noise_amp, netD2, netG2, reals2, fixed_noise2, noise_amp2, opt, depth, writer):
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    reals_shapes2 = [real2.shape for real2 in reals2]
    real2 = reals2[depth]

    # alpha = opt.alpha
    lambda_idt = opt.lambda_idt
    lambda_cyc = opt.lambda_cyc
    lambda_tv = opt.lambda_tv

    ############################
    # define z_opt for training on reconstruction
    ###########################
    if depth == 0:
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            z_opt = reals[0]
            z_opt2 = reals2[0]
        elif opt.train_mode == "animation":
            z_opt = functions.generate_noise([opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                             device=opt.device).detach()
            z_opt2 = functions.generate_noise([opt.nc_im, reals_shapes2[depth][2], reals_shapes2[depth][3]],
                                             device=opt.device).detach()
    else:
        if opt.train_mode == "generation" or opt.train_mode == "animation":
            z_opt0 = functions.generate_noise([opt.nfc,
                                              reals_shapes[depth][2]+opt.num_layer*2,
                                              reals_shapes[depth][3]+opt.num_layer*2],
                                              device=opt.device)

            fixed_noise.append(z_opt0.detach())
            # fakes_shapes = [fake.shape for fake in fixed_noise]
            noise_amp_f = [0.1] * 15
            z_opt = netG(reals, fixed_noise, reals_shapes, noise_amp_f)
            fixed_noise = fixed_noise[: -1]

            z_opt02 = functions.generate_noise([opt.nfc,
                                              reals_shapes2[depth][2]+opt.num_layer*2,
                                              reals_shapes2[depth][3]+opt.num_layer*2],
                                              device=opt.device)

            # fixed_noise2.append(z_opt02.detach())
            # fakes_shapes2 = [fake2.shape for fake2 in fixed_noise2]
            z_opt2 = netG2(reals[1:], z_opt, noise_amp_f)
            fixed_noise2 = fixed_noise2[: -1]
            # criterion = nn.MSELoss()
            # rec_loss = criterion(z_opt1, real)
            #
            # RMSE = torch.sqrt(rec_loss).detach()
            # _noise_amp = opt.noise_amp_init * RMSE
            # noise_amp_f[-1] = _noise_amp
            # fixed_noise.pop()
        else:
            z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                              device=opt.device).detach()
            # 暂时未更新
    fixed_noise.append(z_opt.detach())
    fixed_noise2.append(z_opt2.detach())

    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################
    # setup optimizers for D
    optimizerD = optim.Adam(itertools.chain(netD.parameters(),netD2.parameters()), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]

    for block in netG2.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list2 = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG2.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG2.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list2 += [{"params": netG2.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list2 += [{"params": netG2.tail.parameters(), "lr": opt.lr_g}]

    optimizerG = optim.Adam(itertools.chain(parameter_list, parameter_list2), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    if depth == 0:
        noise_amp.append(1)
    else:
        noise_amp.append(0)
        z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)

        criterion = nn.MSELoss()
        rec_loss = criterion(z_reconstruction, real)

        RMSE = torch.sqrt(rec_loss).detach()
        _noise_amp = opt.noise_amp_init * RMSE
        noise_amp[-1] = _noise_amp

    # start training
    _iter = tqdm(range(opt.niter)) #
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))

        ############################
        # (0) sample noise for unconditional generation
        ###########################
        noise = functions.sample_random_noise(depth, reals_shapes, opt)
        noise2 = functions.sample_random_noise(depth, reals_shapes, opt)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps): #
            # train with real
            # netD.zero_grad()
            optimizerD.zero_grad()

            output = netD(real2)
            errD_real = -output.mean()
            output2 = netD(real)
            errD_real2 = -output2.mean()

            # train with fake
            if j == opt.Dsteps - 1:
                fake = netG(reals, reals_shapes, noise_amp, add_noise=True) # 噪声 + 真实图像
                # fake2, _ = netG(noise2, reals_shapes2, noise_amp2)
            else:
                with torch.no_grad():
                    fake = netG(reals, reals_shapes, noise_amp, add_noise=True)
                    # fake2, _ = netG(noise2, reals_shapes2, noise_amp2)

            output = netD(fake.detach())
            errD_fake = output.mean()
            gradient_penalty = functions.calc_gradient_penalty(netD, real2, fake, opt.lambda_grad, opt.device)

            if j == opt.Dsteps - 1:
                fake2 = netG2(reals2, reals_shapes2, noise_amp2, add_noise=True)
            else:
                with torch.no_grad():
                    fake2 = netG2(reals2, reals_shapes2, noise_amp2, add_noise=True)

            output2 = netD2(fake2.detach())
            errD_fake2 = output2.mean()
            gradient_penalty2 = functions.calc_gradient_penalty(netD2, real, fake2, opt.lambda_grad, opt.device)


            errD_total = errD_real + errD_fake + gradient_penalty + errD_real2 + errD_fake2 + gradient_penalty2
            errD_total.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        optimizerG.zero_grad()
        loss_tv = TVLoss()

        output = netD(fake)
        errG = -output.mean() + lambda_tv * loss_tv(fake)

        output2 = netD2(fake2)
        errG2 = -output2.mean() + lambda_tv * loss_tv(fake2)

        loss = nn.L1Loss() # nn.MSELoss()

        rec = netG(real2, reals_shapes2, noise_amp2) # real
        rec_loss = lambda_idt * loss(rec, real2)
        rec_loss = rec_loss.detach()

        cyc = netG(fake2, reals_shapes2, noise_amp2)
        cyc_loss = lambda_cyc* loss(cyc, real2)
        cyc_loss = cyc_loss.detach()

        rec2 = netG2(real, reals_shapes, noise_amp)
        rec_loss2 = lambda_idt * loss(rec2, real)
        rec_loss2 = rec_loss2.detach()

        cyc2 = netG2(fake, reals_shapes, noise_amp)
        cyc_loss2 = lambda_cyc* loss(cyc2, real)
        cyc_loss2 = cyc_loss2.detach()

        errG_total = errG + rec_loss + errG2 + cyc_loss + cyc_loss2 + rec_loss2
        errG_total.backward()

        for _ in range(opt.Gsteps): # opt.Gsteps
            optimizerG.step()

        ############################
        # (3) Log Results
        ###########################
        if iter % 250 == 0 or iter+1 == opt.niter:
            writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter+1)
            writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter+1)
            writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter+1)
            writer.add_scalar('Loss/train/D/real2/{}'.format(j), -errD_real2.item(), iter+1)
            writer.add_scalar('Loss/train/D/fake2/{}'.format(j), errD_fake2.item(), iter+1)
            writer.add_scalar('Loss/train/D/gradient_penalty2/{}'.format(j), gradient_penalty2.item(), iter+1)

            writer.add_scalar('Loss/train/G/gen', errG.item(), iter+1)
            writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter+1)
            writer.add_scalar('Loss/train/G/cycle', cyc_loss.item(), iter+1)
            writer.add_scalar('Loss/train/G/gen2', errG2.item(), iter+1)
            writer.add_scalar('Loss/train/G/reconstruction2', rec_loss2.item(), iter+1)
            writer.add_scalar('Loss/train/G/cycle2', cyc_loss2.item(), iter+1)

        if iter % 500 == 0 or iter+1 == opt.niter:
            functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter+1), fake.detach())
            functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter+1), rec.detach())
            generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)

        schedulerD.step()
        schedulerG.step()
        # break

    functions.save_networks(netG, netD, z_opt, netG2, netD2, z_opt2, opt)
    return fixed_noise, noise_amp, netG, netD, fixed_noise2, noise_amp2, netG2, netD2


def generate_samples(netG, opt, depth, noise_amp, writer, reals, iter, n=25):
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(depth, reals_shapes, opt)
            sample, _ = netG(noise, reals_shapes, noise_amp)
            all_images.append(sample)
            functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save, idx), sample.detach())

        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[depth].squeeze()
        grid = make_grid(all_images, nrow=min(5, n), normalize=True)
        writer.add_image('gen_images_{}'.format(depth), grid, iter)


def init_G(opt):
    # generator initialization:
    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)

    netG2 = models.GrowingGenerator(opt).to(opt.device)
    netG2.apply(models.weights_init)
    # print(netG)

    return netG, netG2

def init_D(opt):
    #discriminator initialization:
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)

    netD2 = models.Discriminator(opt).to(opt.device)
    netD2.apply(models.weights_init)
    # print(netD)

    return netD, netD2
