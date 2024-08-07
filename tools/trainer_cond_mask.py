import os
import sys

sys.path.extend([sys.path[0][:-4], '/app'])

import time
import copy

import torch
from torch.cuda.amp import GradScaler, autocast

from utils import AverageMeter
from evals.eval_cond import test_psnr_mask, save_image_ddpm_cond, save_image_ddpm_cond_ae, save_image_cond_mask, save_image_ddpm_mask, z_list_gen
from models.ema import LitEma
from einops import rearrange

import nibabel as nib

def latentDDPM(rank, first_stage_model, model, opt, criterion, train_loader, test_loader, scheduler, ema_model=None,
               cond_prob=0.3, logger=None, ddpm_ae=None, ddpm_ae_opt=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # if rank == 0:
    rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    # losses['cont_loss'] = AverageMeter()
    # losses['total_loss'] = AverageMeter()
    check = time.time()

    l1_loss = torch.nn.L1Loss()

    if ema_model == None:
        ema_model = copy.deepcopy(model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200, dtype=torch.int)
        ema_model.eval()

    first_stage_model.eval()
    # ddpm_ae.train()
    model.train()

    # for it, (x, m, cond, _) in enumerate(train_loader):
    for it, (x, m, g, cond, _) in enumerate(train_loader):

        # it = it + 12000

        # x_p_prev = rearrange(torch.zeros(x[0].shape), 'b t c h w -> b c t h w').cuda()
        # m_p_prev = rearrange(torch.zeros(m[0].shape), 'b t c h w -> b c t h w').cuda()

        model.zero_grad()

        # z_list_real = []
        diff_loss = 0.
        for x_idx in range (0, x.__len__()):

            x_p = x[x_idx].to(device)
            m_p = m[x_idx].to(device)
            g_p = g[x_idx].to(device)
            x_p = rearrange(x_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            # x_p = rearrange(x_p / 127.5 - 1., 'b t c h w -> b c t h w').float()  # videos
            m_p = rearrange(m_p + 1e-8, 'b t c h w -> b c t h w').float()
            # m_p = rearrange(2. * m_p - 1., 'b t c h w -> b c t h w').float()
            g_p = rearrange(g_p + 1e-8, 'b t c h w -> b c t h w').float()
            # g_p = rearrange(2. * g_p - 1., 'b t c h w -> b c t h w').float()
            x_p_concat = torch.cat([x_p, g_p], dim=1)
            m_p_concat = torch.cat([m_p, g_p], dim=1)

            cond_p = cond[x_idx].to(device)

            with autocast():
                with torch.no_grad():
                    # z = first_stage_model.encode(x_p_concat, cond_p).detach()
                    z = first_stage_model.extract(x_p_concat, cond_p)
                    z_m = first_stage_model.extract(m_p_concat, cond_p)
                    # z_list_real.append(z)
                    # print(z.shape)
                    # z = first_stage_model.extract_dep(x_p_concat, cond_p)
                    '''
                    print(z.min())
                    print(z.max())
                    out = first_stage_model.decode(z, cond_p).detach()
                    out = out[0].float().detach().cpu().numpy()
                    out = out.squeeze()
                    out = out.swapaxes(0, 2)
                    print(out.shape)
                    out_nii = nib.Nifti1Image(out, None)
                    nib.save(out_nii, os.path.join(logger.logdir, f'out_real_{x_idx}_z_norm.nii.gz'))
  
                    z_out = z[0].float().detach().cpu().numpy()
                    z_out = z_out.reshape([32, 8, 8])
                    z_out_nii = nib.Nifti1Image(z_out, None)
                    nib.save(z_out_nii, os.path.join(logger.logdir, f'z_real_{x_idx}.nii.gz'))
                    '''
                    # z_m = first_stage_model.encode(m_p_concat, cond_p).detach()

            '''
            z_real_enc = ddpm_ae.encode(z)
            z_real_dec = ddpm_ae.decode(z_real_enc)
            ae_loss = l1_loss(z, z_real_dec)

            ae_loss.backward()
            ddpm_ae_opt.step()
            
            ddpm_ae.zero_grad()
            '''
            (loss, t), loss_dict = criterion(z.float(), cond_p.float(), c_m=z_m)

            loss.backward()
            opt.step()

            losses['diffusion_loss'].update(loss.item(), 1)

            # x_p_prev = x_p.clone()
            # m_p_prev = m_p.clone()

            # diff_loss += diff_loss_part

        '''
        diff_loss = diff_loss / float(x.__len__())

        z_list_fake = z_list_gen(rank, ema_model)

        cont_loss = 0.
        for i in range(0, 8):
            value_fake = l1_loss(z_list_fake[i + 1], z_list_fake[i])
            # value_real = l1_loss(z_list_real[i + 1], z_list_real[i])
            # cont_loss += abs(value_fake - value_real)
            cont_loss += value_fake

        # cont_loss.backward()
        # opt.step()

        cont_loss = cont_loss / 8.

        total_loss = (diff_loss + cont_loss)
        total_loss.backward()
        opt.step()

        losses['diffusion_loss'].update(diff_loss.item(), 1)
        losses['cont_loss'].update(cont_loss.item(), 1)
        losses['total_loss'].update(total_loss.item(), 1)
        '''
        # ema model
        if it % 5 == 0:
            ema(model)

        if it % 125 == 0:
            # if logger is not None and rank == 0:
            if logger is not None:
                logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)
                # logger.scalar_summary('train/cont_loss', losses['cont_loss'].average, it)
                # logger.scalar_summary('train/total_loss', losses['total_loss'].average, it)

                '''
                log_('[Time %.3f] [Diffusion %f] [AE %f]' %
                     (time.time() - check, losses['diffusion_loss'].average, losses['ae_loss'].average))
         

                log_('[Time %.3f] [Diffusion %f] [Cont %f] [Total %f]' %
                     (time.time() - check, losses['diffusion_loss'].average, losses['cont_loss'].average, losses['total_loss'].average))
                '''

                log_('[Time %.3f] [Diffusion %f]' %
                     (time.time() - check, losses['diffusion_loss'].average))


                losses = dict()
                losses['diffusion_loss'] = AverageMeter()
                # losses['cont_loss'] = AverageMeter()
                # losses['total_loss'] = AverageMeter()

        if it % 2000 == 0:
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')
            # torch.save(ddpm_ae.state_dict(), rootdir + f'ae_model_{it}.pth')
            ema.copy_to(ema_model)
            torch.save(ema_model.state_dict(), rootdir + f'ema_model_{it}.pth')
            save_image_ddpm_mask(rank, ema_model, first_stage_model, it, test_loader, logger)
            # save_image_ddpm_cond(rank, ema_model, first_stage_model, it, logger)
            # save_image_ddpm_cond_ae(rank, ema_model, first_stage_model, ddpm_ae, it, logger)



def first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # if rank == 0:
    rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['ae_loss'] = AverageMeter()
    losses['L1_loss'] = AverageMeter()
    losses['d_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    l1_criterion = torch.nn.L1Loss()

    if fp:
        # print('fp')
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, 'scaler.pth')))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, 'scaler_d.pth')))
        except:
            print("Fail to load scalers. Start from initial point.")

    model.train()
    disc_start = criterion.discriminator_iter_start

    # for it, (x, m, cond, _) in enumerate(train_loader):
    for it, (x, m, g, cond, _) in enumerate(train_loader):

        it = it + 3750

        # Store previous partitions
        # x_p_prev = rearrange(torch.zeros(x[0].shape), 'b t c h w -> b c t h w').cuda()

        for x_idx in range (0, x.__len__()):

            batch_size = x[x_idx].size(0)
            x_p = x[x_idx].to(device)
            # m_p = m[x_idx].to(device)
            g_p = g[x_idx].to(device)

            # x_p = rearrange(x_p / 127.5 - 1., 'b t c h w -> b c t h w').float()
            # m_p = rearrange(2. * m_p - 1., 'b t c h w -> b c t h w').float()
            # g_p = rearrange(2. * g_p - 1., 'b t c h w -> b c t h w').float()
            x_p = rearrange(x_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            # x_p = rearrange(x_p / 127.5 - 1., 'b t c h w -> b c t h w').float()
            g_p = rearrange(g_p + 1e-8, 'b t c h w -> b c t h w').float()
            # g_p = rearrange(2. * g_p - 1., 'b t c h w -> b c t h w').float()

            x_p_concat = torch.cat([x_p, g_p], dim=1)

            cond_p = cond[x_idx].to(device)

            if not disc_opt:
                with autocast():
                    # x_tilde, vq_loss = model(x)
                    # print(x_p_concat.shape)
                    x_tilde_ra, vq_loss = model(x_p_concat, cond_p)
                    # print(x_tilde_ra.min())
                    # print(x_tilde_ra.max())
                    # print(x_tilde_ra.shape)
                    # print(x_p.shape)
                    # x_tilde_ra = rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size)
                    # if it % accum_iter == 0:
                    model.zero_grad()

                    ae_loss = criterion(vq_loss, x_p, x_tilde_ra,
                                        optimizer_idx=0,
                                        global_step=it)
                    ae_loss = ae_loss / accum_iter

                    l1_loss = l1_criterion(x_p, x_tilde_ra)
                    l1_loss = 10. * l1_loss / accum_iter

                    total_loss = l1_loss
                    # total_loss = l1_loss

                scaler.scale(total_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    scaler.step(opt)
                    scaler.update()

                # print(losses)
                losses['ae_loss'].update(ae_loss.item(), 1)
                losses['L1_loss'].update(l1_loss.item(), 1)

            else:
                if it % accum_iter == 0:
                    criterion.zero_grad()

                with autocast():
                    with torch.no_grad():
                        x_tilde, vq_loss = model(x_p)
                    d_loss = criterion(vq_loss, x_p,
                                       rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                       optimizer_idx=1,
                                       global_step=it)
                    d_loss = d_loss / accum_iter

                scaler_d.scale(d_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler_d.unscale_(d_opt)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(criterion.discriminator_2d.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(criterion.discriminator_3d.parameters(), 1.0)

                    scaler_d.step(d_opt)
                    scaler_d.update()

                losses['d_loss'].update(d_loss.item() * 3, 1)

            if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
                if disc_opt:
                    disc_opt = False
                else:
                    disc_opt = True

            # x_p_prev = x_p.detach().clone()

        if it % 250 == 0:
            psnr = test_psnr_mask(rank, model, test_loader, it, logger)

            if logger is not None:
                logger.scalar_summary('train/ae_loss', losses['ae_loss'].average, it)
                logger.scalar_summary('train/L1_loss', losses['L1_loss'].average, it)
                logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                logger.scalar_summary('test/psnr', psnr, it)

                log_('[Time %.3f] [AELoss %f] [L1Loss %f] [DLoss %f] [PSNR %f]' %
                    (time.time() - check, losses['ae_loss'].average, losses['L1_loss'].average,
                    losses['d_loss'].average, psnr))

                torch.save(model.state_dict(), rootdir + f'model_last.pth')
                torch.save(criterion.state_dict(), rootdir + f'loss_last.pth')
                torch.save(opt.state_dict(), rootdir + f'opt.pth')
                torch.save(d_opt.state_dict(), rootdir + f'd_opt.pth')
                torch.save(scaler.state_dict(), rootdir + f'scaler.pth')
                torch.save(scaler_d.state_dict(), rootdir + f'scaler_d.pth')

            losses = dict()
            losses['ae_loss'] = AverageMeter()
            losses['L1_loss'] = AverageMeter()
            losses['d_loss'] = AverageMeter()

        if it % 1250 == 0:
            save_image_cond_mask(rank, model, test_loader, it, logger)
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')