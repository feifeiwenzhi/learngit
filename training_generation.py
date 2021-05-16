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
from ConSinGAN import imresize
import ConSinGAN.functions as functions
import ConSinGAN.models as models
import copy
import math

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
    fakes, fakes2 = [], []
    in_s, in_s2 = [], []
    for scale_num in range(opt.stop_scale+1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        opt.nfc2 = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc2 = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
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
        print('123345', generator2)

        writer = SummaryWriter(log_dir=opt.outf)
        fixed_noise, noise_amp, generator, d_curr, fixed_noise2, \
        noise_amp2, generator2, d_curr2, in_s, in_s2= \
            train_single_scale(d_curr, generator, reals, fixed_noise, noise_amp, d_curr2, generator2, reals2,
                               fixed_noise2, noise_amp2, opt, scale_num, writer,
                               fakes, fakes2, in_s, in_s2)
        # Gs2.append()
        # Gs.append()
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


# def cycle_rec2(Gs,Gs2,fixed_noise,reals,NoiseAmp,in_s,opt):
#     x_ab = in_s # 打印 in_s 是否不變
#     x_aba = in_s
#     if len(Gs) > 0:
#         count = 0
#         # for G,G2,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Gs2,fixed_noise,reals,reals[1:],NoiseAmp):
#         z = functions.generate_noise([3, Z_opt.shape[2] , Z_opt.shape[3]], device=opt.device)
#         z = z.expand(1, 3, z.shape[2], z.shape[3]) # opt.bsz
#         # z = m_noise(z)
#         x_ab = x_ab[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
#         # x_ab = m_image(x_ab)
#         z_in = noise_amp*z+ real_curr # m_image(real_curr)
#         x_ab = G(z_in.detach(),x_ab)
#
#         x_aba = G2(x_ab,x_aba)
#
#         x_ab = imresize(x_ab.detach(),1/opt.scale_factor,opt)
#         x_ab = x_ab[:,:,0:real_next.shape[2],0:real_next.shape[3]]
#         x_aba = imresize(x_aba.detach(),1/opt.scale_factor,opt)
#         x_aba = x_aba[:,:,0:real_next.shape[2],0:real_next.shape[3]]
#         # count += 1
#     return x_ab, x_aba

def cycle_rec(netG,netG2,fixed_noise,reals,noise_amp,opt, depth, reals_shapes,in_s):
    # netG2,netG,fixed_noise2,reals2,noise_amp2,in_s2,opt, depth 2.1.1
    x_ab = in_s
    x_aba = in_s
    if depth > 0:
        netG_ = copy.deepcopy(netG)
        # netG_.body = netG_.body[:-1]
        netG2_ = copy.deepcopy(netG2)
        netG2_.body = netG2_.body[:-1]
        # for G,G2,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Gs2,fixed_noise,reals,reals[1:],NoiseAmp):
        # z = functions.generate_noise([3, fixed_noise[-2].shape[2], fixed_noise[-2].shape[3]], device=opt.device)
        # z = z.expand(1, 3, z.shape[2], z.shape[3]) # opt.bsz
        # z = m_noise(z)
        # x_ab = x_ab[:,:,0:fixed_noise[-2].shape[2],0:fixed_noise[-2].shape[3]]
        # x_ab = m_image(x_ab)
        # z_in = z + reals[depth - 1] # m_image(real_curr)
        x_ab_e = []
        for k in range(len(netG.body) - 1):
            z_in = functions.sample_random_noise(reals, k, reals_shapes, opt, noise_amp)
            netG_.body = netG.body[:k + 1]
            g_map = netG_(z_in, x_ab[k], reals_shapes)
            x_ab_e.append(g_map.detach())
        # x_ab = netG_(z_in, x_ab, reals_shapes)
        x_aba = netG2_(x_ab_e, x_aba[-1], reals_shapes)
        # x_ab = x_ab.detach()
        x_aba = x_aba.detach()
        if depth == 4:
            opt.scale_factor = 0.455
        if depth == 5:
            opt.scale_factor = 0.6
        x_ab = imresize.imresize(x_ab_e[-1],1/opt.scale_factor,opt) # detach
        x_aba = imresize.imresize(x_aba,1/opt.scale_factor,opt)

        x_ab = x_ab[:,:,0:reals[depth].shape[2],0:reals[depth].shape[3]]
        x_aba = x_aba[:,:,0:reals[depth].shape[2],0:reals[depth].shape[3]]
        # count += 1
        return x_ab, x_aba
    else:
        return x_ab[-1], x_aba[-1]

def draw_concat2(Gs,fixed_noise,reals,NoiseAmp,in_s,mode,opt, reals_shapes):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,fixed_noise,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                # G_z = m_image(G_z)
                z_in = real_curr # m_image(real_curr)
                G_z = G(z_in.detach(),G_z, reals_shapes)
                G_z = imresize(G_z.detach(),1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
    return G_z


def draw_concat(netG, reals, mode, opt, depth, reals_shapes, in_s):
    G_z = in_s[-1]
    if depth > 0:
        if mode == 'rec':
            netG_ = copy.deepcopy(netG)
            netG_.body = netG_.body[:-1]
            # for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,fixed_noise,reals,reals[1:],NoiseAmp):
            G_z = G_z[:, :, 0:reals[depth - 1].shape[2], 0:reals[depth - 1].shape[3]]
            # G_z = m_image(G_z)
            z_in = [] # m_image(real_curr)
            for i in range(depth):
                z_in.append(reals[i].detach())
            G_z = netG_(z_in, G_z, reals_shapes)
            G_z = imresize.imresize(G_z.detach(),1/opt.scale_factor,opt)
            G_z = G_z[:,:,0:reals[depth].shape[2],0:reals[depth].shape[3]]
    return G_z


def train_single_scale(netD, netG, reals, fixed_noise, noise_amp, netD2, netG2,
                       reals2, fixed_noise2, noise_amp2, opt, depth, writer,
                       fakes, fakes2, in_s, in_s2):
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    # reals_shapes2 = [real2.shape for real2 in reals2]
    real2 = reals2[depth]

    # alpha = opt.alpha
    # lambda_idt = opt.lambda_idt
    # lambda_cyc = opt.lambda_cyc
    # lambda_tv = opt.lambda_tv
    lambda_idt = 1
    lambda_cyc = 1
    lambda_tv = 1

    ############################
    # define z_opt for training on reconstruction
    ###########################
    z_opt = functions.generate_noise([3,    # opt.nfc
                                      reals_shapes[depth][2],
                                      reals_shapes[depth][3]],
                                      device=opt.device)

    z_opt2 = functions.generate_noise([3,
                                      reals_shapes[depth][2],
                                      reals_shapes[depth][3]],
                                      device=opt.device)

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
    # if depth - opt.train_depth < 0:
    #     parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    parameter_list += [{"params": netG.head2.parameters(), "lr": opt.lr_g}]
    parameter_list += [{"params": netG.body2.parameters(), "lr": opt.lr_g}]
    parameter_list += [{"params": netG.tail2.parameters(), "lr": opt.lr_g}]

    for block in netG2.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list2 = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG2.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG2.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    # if depth - opt.train_depth < 0:
    #     parameter_list2 += [{"params": netG2.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list2 += [{"params": netG2.tail.parameters(), "lr": opt.lr_g}]
    parameter_list2 += [{"params": netG2.head2.parameters(), "lr": opt.lr_g}]
    parameter_list2 += [{"params": netG2.body2.parameters(), "lr": opt.lr_g}]
    parameter_list2 += [{"params": netG2.tail2.parameters(), "lr": opt.lr_g}]

    optimizerG = optim.Adam(itertools.chain(parameter_list, parameter_list2), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)

    ############################
    # calculate noise_amp       netG(noise, prev, reals_shapes)[-1]
    ###########################
    # if depth == 0:
    #     noise_amp.append(1)
    # else:
    #     noise_amp.append(0)

    # start training
    _iter = tqdm(range(opt.niter))
    loss_print = {}
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))

        ############################
        # (0) sample noise for unconditional generation
        ###########################
        # noise = functions.sample_random_noise(reals, depth, reals_shapes, opt, noise_amp)
        # noise2 = functions.sample_random_noise(reals2, depth, reals_shapes, opt, noise_amp)
        # 1.1.1     1.2.1   2.1.1   2.2.1

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            # netD.zero_grad()
            optimizerD.zero_grad()

            output = netD(real2).to(opt.device)
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)
            loss_print['errD_real'] = errD_real.item()

            output2 = netD2(real).to(opt.device)
            errD_real2 = -output2.mean()
            errD_real2.backward(retain_graph=True)
            loss_print['errD_real2'] = errD_real2.item()

            if (j == 0) & (iter == 0):
                if depth == 0: # 1 opt.bsz   1.1.1
                    noise_amp.append(1)
                    noise_amp2.append(1)
                    prev = torch.full([1,opt.nc_im,reals_shapes[0][2],reals_shapes[0][3]], 0, device=opt.device)
                    in_s.append(prev)
                    # in_s_ = prev

                    prev2 = torch.full([1,opt.nc_im,reals_shapes[0][2],reals_shapes[0][3]], 0, device=opt.device)
                    in_s2.append(prev2)
                    # in_s2_ = prev2

                    c_prev = torch.full([1,opt.nc_im,reals_shapes[depth][2],reals_shapes[depth][3]], 0, device=opt.device)
                    z_prev = torch.full([1,opt.nc_im,reals_shapes[depth][2],reals_shapes[depth][3]], 0, device=opt.device)

                    c_prev2 = torch.full([1 ,opt.nc_im,reals_shapes[depth][2],reals_shapes[depth][3]], 0, device=opt.device)
                    z_prev2 = torch.full([1 ,opt.nc_im,reals_shapes[depth][2],reals_shapes[depth][3]], 0, device=opt.device)

                else:           # 2.1.1
                    # in_s2 = in_s2_
                    # in_s = in_s_
                    prev2, c_prev2 = cycle_rec(netG2,netG,fixed_noise2,reals2,noise_amp2,opt, depth, reals_shapes, in_s2)
                    prev, c_prev = cycle_rec(netG,netG2,fixed_noise,reals,noise_amp,opt, depth, reals_shapes, in_s)
                    z_prev2 = draw_concat(netG, reals2,'rec',opt, depth, reals_shapes, in_s2)
                    z_prev = draw_concat(netG2, reals,'rec',opt, depth, reals_shapes, in_s)
            else:   # 1.1.2     1.1.3       2.1.2       2.1.3   2.2.1
                if len(in_s2) > 1:
                    ins2_index = in_s2[: -1]
                    ins_index = in_s[: -1]
                else:
                    ins2_index = in_s2
                    ins_index = in_s
                prev2, c_prev2 = cycle_rec(netG2,netG,fixed_noise2,reals2,noise_amp2,opt, depth, reals_shapes, ins2_index)
                prev, c_prev = cycle_rec(netG,netG2,fixed_noise,reals,noise_amp,opt, depth, reals_shapes,ins_index)

            if j == 0:  # 1.1.1    1.2.1     2.1.1      2.2.1
                if depth > 0:
                    in_s_ = torch.full([1 ,opt.nc_im,reals_shapes[depth][2],reals_shapes[depth][3]], 0, device=opt.device)
                    in_s2_ = torch.full([1 ,opt.nc_im,reals_shapes[depth][2],reals_shapes[depth][3]], 0, device=opt.device)
                    noise_amp.append(0)
                    noise_amp2.append(0)
                    z_reconstruction = netG2(fixed_noise, in_s_, reals_shapes)
                    z_reconstruction2 = netG(fixed_noise2, in_s2_, reals_shapes)
                    if iter != 0:
                        in_s.pop()
                        in_s2.pop()
                    in_s.append(in_s_)
                    in_s2.append(in_s2_)

                    criterion = nn.MSELoss()
                    rec_loss = criterion(z_reconstruction, real)
                    rec_loss2 = criterion(z_reconstruction2, real2)

                    RMSE = torch.sqrt(rec_loss).detach()
                    RMSE2 = torch.sqrt(rec_loss2).detach()
                    _noise_amp = 0.1 * RMSE # opt.noise_amp_init
                    _noise_amp2 = 0.1 * RMSE2
                    noise_amp[-1] = _noise_amp
                    noise_amp2[-1] = _noise_amp2
                noise = functions.sample_random_noise(reals, depth, reals_shapes, opt, noise_amp)
                noise2 = functions.sample_random_noise(reals2, depth, reals_shapes, opt, noise_amp2)

            # train with fake
            if j == opt.Dsteps - 1:
                fake = netG(noise, prev, reals_shapes)
                fake2 = netG2(noise2, prev2, reals_shapes)
            else:
                with torch.no_grad():
                    fake = netG(noise, prev, reals_shapes)
                    fake2 = netG2(noise2, prev2, reals_shapes)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            loss_print['errD_fake'] = errD_fake.item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real2, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()
            loss_print['gradient_penalty'] = gradient_penalty.item()

            output2 = netD2(fake2.detach())
            errD_fake2 = output2.mean()
            errD_fake2.backward(retain_graph=True)
            loss_print['errD_fake2'] = errD_fake2.item()

            gradient_penalty2 = functions.calc_gradient_penalty(netD2, real, fake2, opt.lambda_grad, opt.device)
            gradient_penalty2.backward()
            loss_print['gradient_penalty2'] = gradient_penalty2.item()

            optimizerD.step()
            # conda activate tui

        if iter != 0:
            fakes.pop()
            fakes2.pop()
        fakes.append(fake)
        fakes2.append(fake2)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        optimizerG.zero_grad()
        loss_tv = TVLoss()

        output = netD(fake)
        errG = -output.mean() + lambda_tv * loss_tv(fake)
        errG.backward(retain_graph=True)
        loss_print['errG'] = errG.item()

        output2 = netD2(fake2)
        errG2 = -output2.mean() + lambda_tv * loss_tv(fake2)
        errG2.backward(retain_graph=True)
        loss_print['errG2'] = errG2.item()

        loss = nn.L1Loss() # nn.MSELoss()

        rec = netG(fixed_noise2, z_prev2, reals_shapes)
        rec_loss = lambda_idt * loss(rec, real2)
        rec_loss.backward(retain_graph=True)
        loss_print['rec_loss'] = rec_loss.item()
        rec_loss = rec_loss.detach()

        cyc = netG(fakes2, c_prev2, reals_shapes)
        cyc_loss = lambda_cyc* loss(cyc, real2)
        cyc_loss.backward(retain_graph=True)
        loss_print['cyc_loss'] = cyc_loss.item()
        cyc_loss = cyc_loss.detach()

        rec2 = netG2(fixed_noise, z_prev, reals_shapes)
        rec_loss2 = lambda_idt * loss(rec2, real)
        rec_loss2.backward(retain_graph=True)
        loss_print['rec_loss2'] = rec_loss2.item()
        rec_loss2 = rec_loss2.detach()

        cyc2 = netG2(fakes, c_prev, reals_shapes)
        cyc_loss2 = lambda_cyc* loss(cyc2, real)
        cyc_loss2.backward(retain_graph=True)
        loss_print['cyc_loss2'] = cyc_loss2.item()
        cyc_loss2 = cyc_loss2.detach()

        for _ in range(opt.Gsteps): # opt.Gsteps
            optimizerG.step()

        ############################
        # (3) Log Results
        ###########################
        if iter % 500 == 0 or iter == (opt.niter-1):
            functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter+1), fake.detach())
            functions.save_image('{}/fake_sample2_{}.jpg'.format(opt.outf, iter+1), fake2.detach())
            # functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter+1), rec.detach())
            # functions.save_image('{}/reconstruction2_{}.jpg'.format(opt.outf, iter+1), rec2.detach())
            # generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)

            log = " Iteration [{}/{}]".format(iter, opt.niter)
            for tag, value in loss_print.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)
        # if iter % 250 == 0 or iter+1 == opt.niter:
        #     writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/real2/{}'.format(j), -errD_real2.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/fake2/{}'.format(j), errD_fake2.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/gradient_penalty2/{}'.format(j), gradient_penalty2.item(), iter+1)
        #
        #     writer.add_scalar('Loss/train/G/gen', errG.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/cycle', cyc_loss.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/gen2', errG2.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/reconstruction2', rec_loss2.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/cycle2', cyc_loss2.item(), iter+1)
        #
        # if iter % 500 == 0 or iter+1 == opt.niter:
        #     functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter+1), fake.detach())
        #     functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter+1), rec.detach())
        #     # generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG, netD, z_opt, netG2, netD2, z_opt2, opt)
    return fixed_noise, noise_amp, netG, netD, fixed_noise2, noise_amp2, netG2, netD2, in_s, in_s2


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
            noise = functions.sample_random_noise(depth, reals_shapes, opt, noise_amp)
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
