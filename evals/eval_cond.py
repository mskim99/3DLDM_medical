import time
import sys;

sys.path.extend(['.', 'src'])
import numpy as np
import torch
from utils import AverageMeter
from torchvision.utils import save_image, make_grid
from einops import rearrange
from losses.ddpm import DDPM

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import os

import torchvision
import PIL

import nibabel as nib

'''
def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print(f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img
    '''

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
                x_p = x[x_idx].float().to(device) / 127.5 - 1
                cond_p = cond[x_idx].to(device)
                # print(x_p.shape)
                # print(x_p_prev.shape)
                x_p_concat = torch.cat([x_p, x_p_prev], dim=2)
                # print(x_p_concat.shape)
                recon, _ = model(rearrange(x_p_concat, 'b t c h w -> b c t h w'), cond_p)

                x_p = x_p.view(batch_size, -1)
                recon = recon.view(batch_size, -1)

                mse = ((x_p * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
                psnr = (-10 * torch.log10(mse)).mean()

                losses['psnr'].update(psnr.item(), batch_size)

    model.train()
    return losses['psnr'].average


def save_image(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    model.eval()
    with torch.no_grad():
        for n, (real, idx) in enumerate(loader):
            if n > 0:
                break
            real = real.float().to(device)
            # cond = cond.to(device)
            fake, _ = model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'))

            # real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))
            # fake = (fake.clamp(-1, 1) + 1) * 127.5

            real = real.type(torch.uint8).cpu().numpy()
            fake = fake.type(torch.uint8).cpu().numpy()

            real = real.squeeze()
            fake = fake.squeeze()

            real = real.swapaxes(1, 3)
            fake = fake.swapaxes(1, 3)

            real_nii = nib.Nifti1Image(real[0, :, :, :], None)
            fake_nii = nib.Nifti1Image(fake[0, :, :, :], None)

            nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}.nii.gz'))
            nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}.nii.gz'))


def save_image_ddpm(rank, ema_model, decoder, it, logger=None, idx_cond=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)

    with torch.no_grad():
        z = diffusion_model.sample(batch_size=8, idx_cond=idx_cond)
        # print(z.shape)
        fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
        # fake = decoder.decode_from_sample(z).clamp(-1, 1).cpu()
        # print(fake.shape)
        fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=8)) * 127.5
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
            real_p_prev = torch.zeros(real[0].shape).to(device)

            for r_idx in range(0, real.__len__()):
                real_p = real[r_idx].float().to(device)
                real_p = real_p / 127.5 - 1
                cond_p = cond[r_idx].to(device)
                real_p_concat = torch.cat([real_p, real_p_prev], dim=2)
                fake, _ = model(rearrange(real_p_concat, 'b t c h w -> b c t h w'), cond_p)

                fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real_p.size(0))

                real_p_prev = real_p

                real_p = real_p.type(torch.uint8).cpu().numpy()
                fake = fake.type(torch.uint8).cpu().numpy()

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

def save_image_ddpm_cond(rank, ema_model, decoder, it, logger=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)

    with torch.no_grad():
        for idx in range(0, 18):
            cond = torch.tensor([idx])
            z = diffusion_model.sample(batch_size=1, cond=cond)
            fake = decoder.decode_from_sample(z).clamp(-1, 1).cpu()
            fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=1)) * 127.5
            fake = fake.type(torch.uint8).cpu().numpy()
            fake = fake.squeeze()
            fake = fake.swapaxes(0, 2)
            fake_nii = nib.Nifti1Image(fake, None)
            nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}.nii.gz'))