import torch
import os

from models.autoencoder.autoencoder_vit_cond import ViTAutoencoder
from models.ddpm.unet import UNetModel, DiffusionWrapper

from utils import Logger

import argparse
from omegaconf import OmegaConf

from tools.dataloader import get_loaders
from evals.eval import save_image_cond

import nibabel as nib
from einops import rearrange

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_config', type=str, default='../configs/autoencoder/base.yaml')
parser.add_argument('--diffusion_config', type=str, default='../configs/latent-diffusion/base.yaml')
parser.add_argument('--batch_size', type=int, default=24)
args = parser.parse_args()
rank = 2

config = OmegaConf.load(args.diffusion_config)
first_stage_config = OmegaConf.load(args.pretrain_config)

args.unetconfig = config.model.params.unet_config
args.res        = first_stage_config.model.params.ddconfig.resolution
args.timesteps  = first_stage_config.model.params.ddconfig.timesteps
args.skip       = first_stage_config.model.params.ddconfig.skip
args.ddconfig   = first_stage_config.model.params.ddconfig
args.embed_dim  = first_stage_config.model.params.embed_dim
args.ddpmconfig = config.model.params
args.cond_model = config.model.cond_model

device = torch.device('cuda', rank)

""" ROOT DIRECTORY """
fn = 'first_stage_main_CHAOS_42_test'
logger = Logger(fn, ask=False)
logger.log(args)
logger.log(f'Log path: {logger.logdir}')
rootdir = logger.logdir

if logger is None:
    log_ = print
else:
    log_ = logger.log

torch.cuda.set_device(rank)
first_stage_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)

unet = UNetModel(**args.unetconfig)
model = DiffusionWrapper(unet).to(device)

first_stage_model_ckpt = torch.load(rootdir + f'model_last.pth', map_location='cuda:2')
first_stage_model.load_state_dict(first_stage_model_ckpt)

# train_loader, test_loader, total_vid = get_loaders(rank, 'CHAOS', args.res, args.timesteps, args.skip, args.batch_size, 1, 42, cond=False)


datas = []
datas_real_concat = []
datas_gen_concat = []

for i in range (0, 18):
    img = nib.load('/data/jionkim/PVDM/test/31_' + str(i) + '_s_16.nii.gz')
    img_data = img.get_fdata()
    img_data = img_data.swapaxes(0, 2)
    img_data = np.expand_dims(img_data, axis=1)
    datas.append(img_data)

print(datas.__len__())

first_stage_model.eval()
with torch.no_grad():
    for idx in range(int(datas.__len__())):

        real = torch.Tensor(datas[idx])
        real = real.float().to(device)
        # print(real.shape)
        real = real.reshape(1, 16, 1, 128, 128)
        cond = torch.tensor([idx], dtype=torch.int64)
        # cond = cond.long().to(device)
        cond = cond.to(device)

        fake, _ = first_stage_model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'), cond)
        # fake, _ = first_stage_model(real / 127.5 - 1, cond)

        # real = rearrange(real, 'b t c h w -> b t h w c') # videos
        fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))
        # fake = (fake.clamp(-1, 1) + 1) * 127.5

        real = real.type(torch.uint8).cpu().numpy()
        fake = fake.type(torch.uint8).cpu().numpy()

        real = real.squeeze()
        fake = fake.squeeze()

        real = real.swapaxes(0, 2)
        fake = fake.swapaxes(0, 2)

        '''
        real_nii = nib.Nifti1Image(real, None)
        fake_nii = nib.Nifti1Image(fake, None)

        nib.save(real_nii, os.path.join(logger.logdir, f'real_{idx}.nii.gz'))
        nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{idx}.nii.gz'))
        '''

        datas_real_concat.append(real)
        datas_gen_concat.append(fake)

datas_real_concat = np.concatenate(datas_real_concat, axis=2)
datas_gen_concat = np.concatenate(datas_gen_concat, axis=2)

real_nii = nib.Nifti1Image(datas_real_concat, None)
fake_nii = nib.Nifti1Image(datas_gen_concat, None)

nib.save(real_nii, os.path.join(logger.logdir, f'real_concat.nii.gz'))
nib.save(fake_nii, os.path.join(logger.logdir, f'generated_concat.nii.gz'))
