import time
import sys; sys.path.extend(['.', 'src'])
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

    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img

def test_psnr(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()
    check = time.time()

    model.eval()
    with torch.no_grad():
        for n, (x, cond, _) in enumerate(loader):
        # for n, (x, _) in enumerate(loader):
            if n > 100:
                break
            batch_size = x.size(0)
            clip_length = x.size(1)
            x = x.float().to(device) / 127.5 - 1
            cond = cond.to(device)
            recon, _ = model(rearrange(x, 'b t c h w -> b c t h w'), cond)

            x = x.view(batch_size, -1)
            recon = recon.view(batch_size, -1)

            mse = ((x * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
            psnr = (-10 * torch.log10(mse)).mean()

            losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average

def test_ifvd(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    real_embeddings = []
    fake_embeddings = []
    fakes = []
    reals = []

    model.eval()
    i3d = load_i3d_pretrained(device)

    with torch.no_grad():
        for n, (real, idx) in enumerate(loader):
            if n > 512:
                break
            batch_size = real.size(0)
            clip_length = real.size(1)
            real = real.float().to(device)
            # print(real.shape)
            # real = torch.cat([real, real, real], dim=2)
            # print(real.shape)
            fake, _ = model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'))

            real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))

            real = real.type(torch.uint8).cpu()
            fake = fake.type(torch.uint8)

            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(fake.cpu().numpy(), i3d=i3d, device=device))
            if len(fakes) < 16:
                reals.append(rearrange(real[0:1], 'b t h w c -> b c t h w'))
                fakes.append(rearrange(fake[0:1], 'b t h w c -> b c t h w'))

    model.train()

    reals = torch.cat(reals)
    fakes = torch.cat(fakes)

    # if rank == 0:
    real_vid = save_image_grid(reals.cpu().numpy(), os.path.join(logger.logdir, "real.gif"), drange=[0, 255], grid_size=(4,4))
    fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,4))

    if it == 0:
        real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
        logger.video_summary('real', real_vid, it)

    fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
    logger.video_summary('recon', fake_vid, it)

    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    
    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()


def test_fvd_ddpm(rank, ema_model, decoder, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    cond_model = ema_model.diffusion_model.cond_model

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)
    real_embeddings = []
    fake_embeddings = []
    pred_embeddings = []

    reals = []
    fakes = []
    predictions = []

    i3d = load_i3d_pretrained(device)

    if cond_model:
        with torch.no_grad():        
            for n, (x, _) in enumerate(loader):
                k = min(4, x.size(0))
                if n >= 4:
                    break
                    # x = torch.cat([x,zeros], dim=0)
                c, real = torch.chunk(x[:k], 2, dim=1)
                c = decoder.extract(rearrange(c / 127.5 - 1, 'b t c h w -> b c t h w').to(device).detach())
                z = diffusion_model.sample(batch_size=k, cond=c)
                pred = decoder.decode_from_sample(z).clamp(-1,1).cpu()
                pred = (1 + rearrange(pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
                pred = pred.type(torch.uint8)
                pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))

                real = rearrange(real, 'b t c h w -> b t h w c')
                real = real.type(torch.uint8)
                real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

                if len(predictions) < 4:
                    reals.append(rearrange(x[:k].type(torch.uint8), 'b t c h w -> b c t h w'))
                    predictions.append(torch.cat([rearrange(x[:k,:x.size(1)//2].type(torch.uint8), 'b t c h w -> b c t h w').type(torch.uint8), 
                                                  rearrange(pred, 'b t h w c -> b c t h w')], dim=2))


            for i in range(4):
                print(i)
                z = diffusion_model.sample(batch_size=k)
                fake = decoder.decode_from_sample(z).clamp(-1,1).cpu()
                fake = (rearrange(fake, '(b t) c h w -> b t h w c', b=k)+1) * 127.5
                fake = fake.type(torch.uint8)
                fake_embeddings.append(get_fvd_logits(fake.numpy(), i3d=i3d, device=device))

                if len(fakes) < 4:
                    fakes.append(rearrange(fake, 'b t h w c -> b c t h w'))

        reals = torch.cat(reals)
        fakes = torch.cat(fakes)
        predictions = torch.cat(predictions)

        real_embeddings = torch.cat(real_embeddings)
        fake_embeddings = torch.cat(fake_embeddings)

        # if rank == 0:
        real_vid = save_image_grid(reals.cpu().numpy(), os.path.join(logger.logdir, f'real_{it}.gif'), drange=[0, 255], grid_size=(k,4))
        real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
        fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(k,4))
        fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
        pred_vid = save_image_grid(predictions.cpu().numpy(), os.path.join(logger.logdir, f'predicted_{it}.gif'), drange=[0, 255], grid_size=(k,4))
        pred_vid = np.expand_dims(pred_vid,0).transpose(0, 1, 4, 2, 3)

        logger.video_summary('real', real_vid, it)
        logger.video_summary('unconditional', fake_vid, it)
        logger.video_summary('prediction', pred_vid, it)

    else:
        with torch.no_grad():        
            for n, (real, _) in enumerate(loader):
                if n >= 4:
                    break
                real = rearrange(real, 'b t c h w -> b t h w c')
                real = real.type(torch.uint8).numpy()
                real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))

            for i in range(4):
                print(i)
                z = diffusion_model.sample(batch_size=4)
                fake = decoder.decode_from_sample(z).clamp(-1,1).cpu()
                fake = (1+rearrange(fake, '(b t) c h w -> b t h w c', b=4)) * 127.5
                fake = fake.type(torch.uint8)
                fake_embeddings.append(get_fvd_logits(fake.numpy(), i3d=i3d, device=device))

                if len(fakes) < 4:
                    fakes.append(rearrange(fake, 'b t h w c -> b c t h w'))

        fakes = torch.cat(fakes)

        real_embeddings = torch.cat(real_embeddings)
        fake_embeddings = torch.cat(fake_embeddings)

        # if rank == 0:
        fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,4))
        fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
        logger.video_summary('unconditional', fake_vid, it)

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()


def save_image(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    model.eval()
    with torch.no_grad():
        for n, (real, idx) in enumerate(loader):
            if n > 0:
                break
            real = real.float().to(device)
            cond = cond.to(device)
            fake, _ = model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'), cond)

            # real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))
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


def save_image_ddpm(rank, ema_model, decoder, it, logger=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)

    with torch.no_grad():
        z = diffusion_model.sample(batch_size=1)
        fake = decoder.decode_from_sample(z).clamp(-1, 1).cpu()
        fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=1)) * 127.5
        fake = fake.type(torch.uint8).cpu().numpy()
        fake = fake.squeeze()
        fake = fake.swapaxes(0, 2)
        fake_nii = nib.Nifti1Image(fake, None)
        nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}.nii.gz'))

def save_image_cond(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    gen_idx_list = []

    model.eval()
    with torch.no_grad():
        for n, (real, cond, idx) in enumerate(loader):

            if gen_idx_list.__len__() >= 18:
                break

            if all(ele in gen_idx_list for ele in cond):
                continue

            real = real.float().to(device)
            cond = cond.to(device)
            fake, _ = model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'), cond)

            # real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))
            # fake = (fake.clamp(-1, 1) + 1) * 127.5

            real = real.type(torch.uint8).cpu().numpy()
            fake = fake.type(torch.uint8).cpu().numpy()

            real = real.squeeze()
            fake = fake.squeeze()

            real = real.swapaxes(1, 3)
            fake = fake.swapaxes(1, 3)

            for b_idx in range (0, real.shape[0]):
                if not cond[b_idx] in gen_idx_list:
                    real_nii = nib.Nifti1Image(real[b_idx, :, :, :], None)
                    fake_nii = nib.Nifti1Image(fake[b_idx, :, :, :], None)

                    nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{cond[b_idx]}.nii.gz'))
                    nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{cond[b_idx]}.nii.gz'))

                    gen_idx_list.append(cond[b_idx].item())