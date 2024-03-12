import os
import json

import torch

from tools.trainer import latentDDPM
from tools.dataloader import get_loaders
from tools.scheduler import LambdaLinearScheduler
from models.autoencoder.autoencoder_vit import ViTAutoencoder
from models.ddpm.unet import UNetModel, DiffusionWrapper
from losses.ddpm import DDPM

import copy
from utils import file_name, Logger, download

import argparse
from omegaconf import OmegaConf

from evals.eval import save_image_ddpm

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_config', type=str, default='../configs/autoencoder/base.yaml')
parser.add_argument('--diffusion_config', type=str, default='../configs/latent-diffusion/base.yaml')
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
fn = 'ddpm_main_CHAOS_42_5'
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

# if rank == 0:
first_stage_model_ckpt = torch.load(rootdir + f'model_last.pth', map_location='cuda:2')
first_stage_model.load_state_dict(first_stage_model_ckpt)

model_ckpt = torch.load(rootdir + f'ddpm_model.pth', map_location='cuda:2')
model.load_state_dict(model_ckpt)
ema_model = copy.deepcopy(model)

save_image_ddpm(rank, ema_model, first_stage_model, 0, logger)
