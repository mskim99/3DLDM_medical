import torch
import torch.nn as nn

from models.autoencoder.vit_modules_spade import Encoder, Decoder


class ViTAutoencoder_SPADE(nn.Module):
    def __init__(self,
                 embed_dim,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(ch=64, ch_mult=(1,2,4,8), num_res_blocks=2, dropout=0.0, resamp_with_conv=True, in_channels=3,
                               resolution=64, z_channels=64) # ch, res, z_channels = 128
        self.decoder = Decoder(ch=64, out_ch=1, ch_mult=(1,2,4,8), num_res_blocks=2, dropout=0.0, resamp_with_conv=True,
                               in_channels=1, resolution=64, z_channels=64) # ch, res, z_channels = 128

        self.quant_conv = torch.nn.Conv3d(64, embed_dim, 1) # 1st = z_channels
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, 64, 1) # 1st = z_channels

    def encode(self, x, cond):
        h = self.encoder(x, cond)
        h = self.quant_conv(h)
        h = torch.tanh(h)
        h = self.post_quant_conv(h)

        return h

    def extract(self, x, cond):
        h = self.encoder(x, cond)
        h = self.quant_conv(h)
        h = torch.tanh(h)
        return h

    def extract_dep(self, x, cond):
        h = self.encoder.extract_dep(x, cond)
        return h

    def decode(self, quant, cond):
        dec = self.decoder(quant, cond)
        return dec

    def decode_from_sample(self, quant, cond):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, cond)
        return dec

    def decode_from_sample_dep(self, quant, cond):
        quant = self.encoder.channel_conv_out(quant)
        dec = self.decoder(quant, cond)
        return dec

    def forward(self, input, cond):
        quant = self.encode(input, cond)
        dec = self.decode(quant, cond)
        return dec, 0.