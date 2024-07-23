import torch
import torch.nn as nn

from models.autoencoder.vit_modules_spade import Encoder, Decoder

import nibabel as nib

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta, remap=None, sane_index_shape=False):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class ViTAutoencoder_SPADE(nn.Module):
    def __init__(self,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.encoder = Encoder(ch=128, ch_mult=(1,2,4,8), num_res_blocks=2, dropout=0.0, resamp_with_conv=True, in_channels=3,
                               resolution=128, z_channels=3)
        self.decoder = Decoder(ch=128, out_ch=1, ch_mult=(1,2,4,8), num_res_blocks=2, dropout=0.0, resamp_with_conv=True,
                               in_channels=1, resolution=128, z_channels=3)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(3, embed_dim, 1) # z_channels = 3
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 3, 1) # z_channels = 3

    def encode(self, x, cond):
        h = self.encoder(x, cond)
        # h = self.quant_conv(h)
        # quant, _, _ = self.quantize(h)
        # return quant, emb_loss, info

        return h

    def extract(self, x, cond):
        h = self.encoder.extract(x, cond)
        return h

    def decode(self, quant, cond):
        # quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, cond)
        return dec

    def decode_from_sample(self, quant, cond):
        quant = self.encoder.channel_conv_out(quant)
        dec = self.decoder(quant, cond)
        return dec


    def forward(self, input, cond):
        # quant, diff, (_,_,ind) = self.encode(input)
        quant = self.encode(input, cond)
        dec = self.decode(quant, cond)
        # return dec, diff
        return dec, 0.