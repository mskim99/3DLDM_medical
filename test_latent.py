import sys
sys.path.extend(['.'])

import argparse

import torch
from omegaconf import OmegaConf

from utils import set_random_seed
from tools.dataloader import get_loaders

import warnings
from models.autoencoder.autoencoder_vit_cond import ViTAutoencoder

from einops import rearrange

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='main', help='experiment identifier')

""" Args about Data """
parser.add_argument('--data', type=str, default='UCF101')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--ds', type=int, default=4)

""" Args about Model """
parser.add_argument('--pretrain_config', type=str, default='configs/autoencoder/autoencoder_kl_f4d6_res128.yaml')

# for GAN resume
parser.add_argument('--first_stage_folder', type=str, default='',
                    help='the folder of first stage experiment before GAN')

# for diffusion model path specification
parser.add_argument('--first_model', type=str, default='', help='the path of pretrained model')
parser.add_argument('--scale_lr', action='store_true')


def test_latent():
    """ Additional args ends here. """
    args = parser.parse_args()
    """ FIX THE RANDOMNESS """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.n_gpus = 1

    # init and save configs
    first_stage_config = OmegaConf.load(args.pretrain_config)

    args.res = first_stage_config.model.params.ddconfig.resolution
    args.timesteps = first_stage_config.model.params.ddconfig.timesteps
    args.skip = first_stage_config.model.params.ddconfig.skip
    args.ddconfig = first_stage_config.model.params.ddconfig
    args.embed_dim = first_stage_config.model.params.embed_dim

    device = torch.device('cuda', 3)

    train_loader, test_loader, total_vid = get_loaders(4, args.data, args.res, args.timesteps, args.skip,
                                                       args.batch_size, args.n_gpus, args.seed, cond=False)


    torch.cuda.set_device(3)
    first_stage_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)

    # if rank == 0:
    first_stage_model_ckpt = torch.load(args.first_model, map_location='cuda:3')
    first_stage_model.load_state_dict(first_stage_model_ckpt)
    del first_stage_model_ckpt

    gt_p = []
    with torch.no_grad():
        for it, (x, cond, _) in enumerate(train_loader):

            # x_p_prev = rearrange(torch.zeros(x[0].shape), 'b t c h w -> b c t h w').cuda()
            for x_idx in range (0, x.__len__()):

                x_p = x[x_idx].to(device)
                x_p = rearrange(x_p / 127.5 - 1, 'b t c h w -> b c t h w').float()  # videos
                cond_p = cond[x_idx].to(device)
                # x_p_concat = torch.cat([x_p, x_p_prev], dim=1)

                z = first_stage_model.extract(x_p, cond_p).detach()
                gt_p.append(z)

                # x_p_prev = x_p.clone()

            gt_diff_sum = 0
            for i in range(len(gt_p) - 1):
                gt_diff = abs(gt_p[i] - gt_p[i + 1]).mean()
                gt_diff_sum += gt_diff
                print(str(i) + ' : ' + str(gt_diff))

            print('MEAN : ' + str(gt_diff_sum / len(gt_p)))

            gt_p = []

if __name__ == '__main__':
    test_latent()