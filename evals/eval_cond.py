import sys

import numpy as np

sys.path.extend(['.', 'src'])
import torch
from utils import AverageMeter
from einops import rearrange
from losses.ddpm_mask import DDPM

import os
import nibabel as nib

def z_list_gen(rank, ema_model):

    device = torch.device('cuda', rank)
    l1_loss = torch.nn.L1Loss()
    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=10,
                           w=0.).to(device)

    z_list = []
    # cont_loss_value = 0.
    for idx in range(0, 9):
        idx_cond = torch.tensor([idx]).to(device)
        z = diffusion_model.sample_3d(batch_size=1, channels=32, idx_cond=idx_cond, tqdm=False)
        z_list.append(z)

    '''
    for i in range (0, 8):
        value = l1_loss(z_list[i+1], z_list[i])
        cont_loss_value += value

    cont_loss_value = cont_loss_value / 8.
    return cont_loss_value
    '''

    return z_list

def test_psnr(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()

    model.eval()
    with torch.no_grad():
        for n, (x, cond, _) in enumerate(loader):

            if n > 12:
                break

            # Store previous partitions
            x_p_prev = torch.zeros(x[0].shape).cuda()

            for x_idx in range(0, x.__len__()):
                batch_size = x[x_idx].size(0)
                x_p = x[x_idx].float().to(device)
                cond_p = cond[x_idx].to(device)
                # print(x_p.shape)
                # print(x_p_prev.shape)
                # x_p_concat = torch.cat([x_p / 127.5 - 1, x_p_prev], dim=2)
                # print(x_p_concat.shape)
                # recon, _ = model(rearrange(x_p_concat, 'b t c h w -> b c t h w'), cond_p)
                recon, _ = model(rearrange(x_p / 127.5 - 1, 'b t c h w -> b c t h w'), cond_p)

                # x_p_prev = x_p.clone().detach() / 127.5 - 1

                x_p = x_p.view(batch_size, -1)
                recon = recon.view(batch_size, -1)

                mse = ((x_p * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
                psnr = (-10 * torch.log10(mse)).mean()

                losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average


def test_psnr_mask(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()

    model.eval()
    with torch.no_grad():
        for n, (x, m, g, cond, _) in enumerate(loader):

            if n > 12:
                break

            for x_idx in range(0, x.__len__()):
                batch_size = x[x_idx].size(0)
                x_p = x[x_idx].float().to(device)
                # m_p = m[x_idx].float().to(device)
                g_p = m[x_idx].float().to(device)
                x_p = rearrange(x_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
                # x_p = rearrange(x_p / 127.5 - 1., 'b t c h w -> b c t h w').float()
                # m_p = rearrange(2. * m_p - 1., 'b t c h w -> b c t h w').float()
                g_p = rearrange(g_p + 1e-8, 'b t c h w -> b c t h w').float()
                # g_p = rearrange(2. * g_p - 1., 'b t c h w -> b c t h w').float()
                x_p_concat = torch.cat([x_p, g_p], dim=1)
                cond_p = cond[x_idx].to(device)
                recon, _ = model(x_p_concat, cond_p)

                # x_p_prev = x_p.clone().detach() / 127.5 - 1

                x_p = x_p.view(batch_size, -1)
                recon = recon.view(batch_size, -1)

                mse = ((x_p * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
                psnr = (-10 * torch.log10(mse)).mean()

                losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average


def save_image_ddpm(rank, ema_model, decoder, it, logger=None, idx_cond=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)

    with torch.no_grad():
        z = diffusion_model.sample(batch_size=16, idx_cond=idx_cond)
        # print(z.shape)
        fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
        # fake = decoder.decode_from_sample(z).clamp(-1, 1).cpu()
        # print(fake.shape)
        fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=16)) * 127.5
        # print(fake.shape)
        fake = fake[0].type(torch.uint8).cpu().numpy()
        # print(fake.shape)
        # for s_i in range (0, 8):
        fake = fake.squeeze()
        # print(fake.shape)
        fake = fake.swapaxes(0, 2)
        # print(fake.shape)
        fake_nii = nib.Nifti1Image(fake, None)
        nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))


def save_image_cond(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    model.eval()
    with torch.no_grad():
        for _, (real, cond, idx) in enumerate(loader):
        # for _, (real, cond, grad, idx) in enumerate(loader):

            # Store previous partitions
            # real_p_prev = torch.zeros(real[0].shape).to(device)

            for r_idx in range(0, real.__len__()):
                real_p = real[r_idx].float().to(device)
                cond_p = cond[r_idx].to(device)
                # grad_p = grad[r_idx].to(device)
                # real_p = rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w')
                # mask_p = rearrange(2. * mask_p - 1., 'b t c h w -> b c t h w')
                # grad_p = rearrange(2. * grad_p - 1., 'b t c h w -> b c t h w')
                # real_p_concat = torch.cat([real_p, grad_p], dim=1)
                # fake, _ = model(real_p, cond_p)
                # real_p_concat = torch.cat([real_p / 127.5 - 1, real_p_prev], dim=2)
                # fake, _ = model(rearrange(real_p_concat, 'b t c h w -> b c t h w'), cond_p)
                # fake, _ = model(rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w'), cond_p)
                fake, _ = model(rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w'), cond_p)
                # fake, _ = model(rearrange(real_p / 255. + 1e-8, 'b t c h w -> b c t h w'), cond_p)

                # fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, '(b t) c h w -> b c t h w', b=real_p.size(0))
                # fake = rearrange((fake.clamp(0, 1) + 1) * 255., '(b t) c h w -> b c t h w', b=real_p.size(0))
                fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, 'b c t h w -> b t c h w', b=real_p.size(0))
                # fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, '(b t nc) c h w -> b nc (c t) h w', b=real_p.size(0), nc=1)

                # real_p_prev = real_p.clone().detach() / 127.5 - 1

                real_p = real_p.type(torch.uint8).cpu().numpy()
                fake = fake.type(torch.uint8).cpu().numpy()

                # real_p = real_p.cpu().numpy()
                # fake = fake.cpu().numpy()

                real_p = real_p.squeeze()
                fake = fake.squeeze()

                real_p = real_p.swapaxes(1, 3)
                fake = fake.swapaxes(1, 3)

                # print(real_p.shape)
                # print(fake.shape)

                real_nii = nib.Nifti1Image(real_p[0, :, :, :], None)
                fake_nii = nib.Nifti1Image(fake[0, :, :, :], None)

                nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{cond[r_idx][0]}.nii.gz'))
                nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{cond[r_idx][0]}.nii.gz'))

            print('eval finished')
            return

def save_image_cond_mask(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    model.eval()
    with torch.no_grad():
        # for _, (real, mask, cond, idx) in enumerate(loader):
        for _, (real, mask, grad, cond, idx) in enumerate(loader):

            # Store previous partitions
            # real_p_prev = torch.zeros(real[0].shape).to(device)

            for r_idx in range(0, real.__len__()):
                real_p = real[r_idx].float().to(device)
                # mask_p = mask[r_idx].float().to(device)
                grad_p = grad[r_idx].float().to(device)
                # real_p = real_p / 127.5 - 1
                cond_p = cond[r_idx].to(device)
                # real_p = rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w')
                real_p = rearrange(real_p / 255. + 1e-8, 'b t c h w -> b c t h w')
                # mask_p = rearrange(2. * mask_p - 1., 'b t c h w -> b c t h w')
                # grad_p = rearrange(2. * grad_p - 1., 'b t c h w -> b c t h w')
                grad_p = rearrange(grad_p + 1e-8, 'b t c h w -> b c t h w')
                real_p_concat = torch.cat([real_p, grad_p], dim=1)
                # fake, _ = model(rearrange(real_p_concat, 'b t c h w -> b c t h w'), cond_p)
                fake, _ = model(real_p_concat, cond_p)

                # fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, 'b c t h w -> b t c h w', b=real_p.size(0))
                fake = rearrange((fake.clamp(0, 1)) * 255., 'b c t h w -> b t c h w', b=real_p.size(0))
                # real_p = rearrange((real_p.clamp(-1, 1) + 1) * 127.5, 'b t c h w -> b t h w c', b=real_p.size(0))
                # real_p = (real_p.clamp(-1, 1) + 1) * 127.5
                real_p = (real_p.clamp(0, 1)) * 255.
                # real_p_prev = real_p.clone().detach() / 127.5 - 1

                real_p = real_p.type(torch.uint8).cpu().numpy()
                fake = fake.type(torch.uint8).cpu().numpy()

                real_p = real_p.squeeze()
                fake = fake.squeeze()

                real_p = real_p.swapaxes(1, 3)
                fake = fake.swapaxes(1, 3)

                real_nii = nib.Nifti1Image(real_p[0, :, :, :], None)
                fake_nii = nib.Nifti1Image(fake[0, :, :, :], None)

                nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{cond[r_idx][0]}.nii.gz'))
                nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{cond[r_idx][0]}.nii.gz'))

            print('eval finished')
            return

def save_image_ddpm_cond(rank, ema_model, decoder, it, logger=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)


    with torch.no_grad():

        # init_context = None
        # z_prev = None
        for idx in range(0, 9):
            idx_cond = torch.tensor([idx]).to(device)
            # z = diffusion_model.sample(batch_size=1, idx_cond=idx_cond, init_context=init_context)
            z = diffusion_model.sample_3d(batch_size=1, channels=32, idx_cond=idx_cond)
            '''
            z_out = z.float().detach().cpu().numpy()
            z_out = z_out.reshape([32, 8, 8])
            z_out_nii = nib.Nifti1Image(z_out, None)
            nib.save(z_out_nii, os.path.join(logger.logdir, f'z_fake_{it}_{idx_cond[0]}.nii.gz'))
            '''
            # print(z.shape)
            z = z.view(1, 32, 1, 8, 8)
            z[:, :, :, 0:1, 0:1] = 0.
            # print('eval')
            # print(z.shape)
            # if z_prev is None:
                # z_prev = z.clone()
            #             # z_concat = torch.cat([z, z_prev], dim=1)
            # print(z_concat.shape)
            # print(z.shape)
            # init_context = z
            # fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
            fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(0, 1).cpu()

            # fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
            # fake = (1 + rearrange(fake, 'b t c h w -> b t h w c', b=1)) * 127.5
            fake = (rearrange(fake, 'b t c h w -> b t h w c', b=1)) * 255.
            # fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=1)) * 127.5
            fake = fake.type(torch.uint8).cpu().numpy()
            fake = fake.squeeze()
            # print(fake.shape)
            # fake = fake.swapaxes(0, 2)
            fake_nii = nib.Nifti1Image(fake, None)
            nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))
            # z_prev = z.clone()


def save_image_ddpm_cond_ae(rank, ema_model, decoder, ae_model, it, logger=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)


    with torch.no_grad():

        for idx in range(0, 9):
            idx_cond = torch.tensor([idx]).to(device)
            z = diffusion_model.sample_3d(batch_size=1, channels=32, idx_cond=idx_cond)
            # z = z.view(1, 32, 2, 16, 16) # res 128
            z = z.view(1, 32, 2, 16, 16) # res 64
            # z_dec = ae_model.decode(z)
            fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(0, 1).cpu()

            fake = (rearrange(fake, 'b t c h w -> b t h w c', b=1)) * 255.
            # print(fake.shape)
            fake = fake.type(torch.uint8).cpu().numpy()
            fake = fake.squeeze()
            fake_nii = nib.Nifti1Image(fake, None)
            nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))


def save_image_ddpm_mask(rank, ema_model, first_stage_model, it, loader, logger=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)


    with torch.no_grad():

        # for _, (_, mask, cond, _) in enumerate(loader):
        for _, (_, mask, grad, cond, _) in enumerate(loader):

            for idx in range(0, mask.__len__()):
                idx_cond = torch.tensor([idx]).to(device)
                m_p = mask[idx].to(device)
                g_p = grad[idx].to(device)
                # m_p = m_p[0].unsqueeze(0)
                # g_p = g_p[0].unsqueeze(0)
                m_p = rearrange(m_p + 1e-8, 'b t c h w -> b c t h w', b=8).float()
                g_p = rearrange(g_p + 1e-8, 'b t c h w -> b c t h w', b=8).float()
                m_p_concat = torch.cat([m_p, g_p], dim=1)
                cond_p = cond[idx].to(device)
                # cond_p = cond_p[0].unsqueeze(0)

                # print(m_p_concat.shape)
                # print(cond_p.shape)

                z_m = first_stage_model.encode(m_p_concat, cond_p)
                z_mask = z_m[0].view(1, 3, 2, 16, 16)
                # print(z_mask.shape)
                cond_p = cond_p[0].unsqueeze(0)
                z = diffusion_model.sample_3d(batch_size=1, channels=3, idx_cond=cond_p, mask=z_mask)
                z = z.view(1, 3, 2, 16, 16)

                # Debugging
                '''
                z_out = z.float().detach().cpu().numpy()
                z_out = z_out.reshape([6, 16, 16])
                z_out_nii = nib.Nifti1Image(z_out, None)
                nib.save(z_out_nii, os.path.join(logger.logdir, f'z_fake.nii.gz'))
                '''

                fake = first_stage_model.decode(z, cond=idx_cond).clamp(0, 1).cpu()
                # fake = first_stage_model.decode_from_sample(z, cond=idx_cond).cpu()
                # fake = 2. * ((fake - fake.min()) / (fake.max() - fake.min())) - 1.
                # print(fake.shape)
                fake = rearrange(fake, 'b t c h w -> b t h w c', b=1) * 255.
                mask_out = rearrange(m_p[0].unsqueeze(0), 'b t c h w -> b t h w c', b=1)
                # fake = fake[1]
                # print(mask.shape)
                # print(mask.min())
                # print(mask.max())
                fake = fake.type(torch.uint8).cpu().numpy()
                mask_out = mask_out.type(torch.uint8).cpu().numpy()
                fake = fake.squeeze()
                mask_out = mask_out.squeeze()
                # fake = fake.swapaxes(0, 2)
                # print(fake.shape)

                fake_nii = nib.Nifti1Image(fake, None)
                mask_nii = nib.Nifti1Image(mask_out, None)
                nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))
                nib.save(mask_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}_mask.nii.gz'))

                m_p_prev = m_p.clone()

            break