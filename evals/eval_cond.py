import sys

sys.path.extend(['.', 'src'])
import torch
from utils import AverageMeter
from einops import rearrange
from losses.ddpm_cond import DDPM

import os
import nibabel as nib


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
            # x_p_prev = torch.zeros(x[0].shape).cuda()

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

            # Store previous partitions
            # real_p_prev = torch.zeros(real[0].shape).to(device)

            for r_idx in range(0, real.__len__()):
                real_p = real[r_idx].float().to(device)
                # real_p = real_p / 127.5 - 1
                cond_p = cond[r_idx].to(device)
                # real_p_concat = torch.cat([real_p / 127.5 - 1, real_p_prev], dim=2)
                # fake, _ = model(rearrange(real_p_concat, 'b t c h w -> b c t h w'), cond_p)
                fake, _ = model(rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w'), cond_p)

                fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real_p.size(0))

                # real_p_prev = real_p.clone().detach() / 127.5 - 1

                real_p = real_p.type(torch.uint8).cpu().numpy()
                fake = fake.type(torch.uint8).cpu().numpy()

                # real_p = real_p.cpu().numpy()
                # fake = fake.cpu().numpy()

                real_p = real_p.squeeze()
                fake = fake.squeeze()

                real_p = real_p.swapaxes(1, 3)
                fake = fake.swapaxes(1, 3)

                real_nii = nib.Nifti1Image(real_p[1, :, :, :], None)
                fake_nii = nib.Nifti1Image(fake[1, :, :, :], None)

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

        init_context = None
        for idx in range(0, 9):
            idx_cond = torch.tensor([idx]).to(device)
            z = diffusion_model.sample(batch_size=1, idx_cond=idx_cond, init_context=init_context)
            # print(z.shape)
            init_context = z
            fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
            fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=1)) * 127.5
            fake = fake.type(torch.uint8).cpu().numpy()
            fake = fake.squeeze()
            fake = fake.swapaxes(0, 2)
            fake_nii = nib.Nifti1Image(fake, None)
            nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))