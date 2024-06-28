import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from math import log, pi



def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rot_emb(q, k, rot_emb):
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device = device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model


    def forward(self):
        pos = torch.arange(self.max_len)[:, None]

        input_tensor = torch.Tensor([10000])
        angles = pos / torch.pow(input_tensor, (2 * torch.arange(self.d_model) // 2) / float(self.d_model))

        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        return angles

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., label_conc=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + label_conc, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim + label_conc)
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.n_channels = 1
        self.down_num = 2
        self.inc = DoubleConv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 32)
    def forward(self, x):
        # features = []
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        # features.append(x1)
        x2 = self.down1(x1)
        # print(x2.shape)
        # features.append(x2)
        feats = self.down2(x2)
        # print(feats.shape)
        # features.append(feats)
        '''
        feats_down = feats
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
            '''
        # return feats, features[::-1]
        return feats


class Embeddings_3D(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # self.cnn_encoder = CNNEncoder()
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x.shape)
        # x = self.cnn_encoder(x)
        # print(x.shape)
        x = self.patch_embeddings(x)
        # print(x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

# attention
def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        label_conc=10,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim + label_conc, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim + label_conc),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q, k = apply_rot_emb(q, k, rot_emb)

        # expand cls token keys and values across time or space and concat
        # attention
        out = attn(q, k, v, mask = mask)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

# main classes

class TimeSformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        num_frames = 16,
        image_size = 128,
        patch_size = 8,
        channels = 1,
        depth = 8,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True,
        nclass=18,
        label_conc=10,
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim * 8, dim * 8)
        self.positional_embedding = PositionalEmbedding(4, 4096)
        self.label_embedding = nn.Embedding(nclass, num_positions)
        # self.embedding_3d = Embeddings_3D(1, dim, [64, 64, 64], int(patch_size / 2), 0.0)
        self.final_fc = nn.Linear(dim + label_conc, dim)

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions, dim * 2)

        self.layers = nn.ModuleList([])
        for _ in range(depth): # First parameter : dim
            ff = FeedForward(dim, dropout = ff_dropout, label_conc=label_conc)
            time_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, label_conc=label_conc)
            spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, label_conc=label_conc)

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim + label_conc, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

    def forward(self, video, cond, lc, frame_mask = None):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)
        hp, wp, fp = (h // p), (w // p), (f // p)
        n = hp * wp

        video = rearrange(video, 'b f c h w -> b h c f w') # (xy)
        x = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p) # xyz patch extraction
        x = self.to_patch_embedding(x)

        cond = self.label_embedding(cond)
        cond = repeat(cond, 'm n -> m n k', k=lc)

        x = torch.cat([x, cond], 2)

        # print(x.shape)

        # positional embedding
        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(fp, device = device)
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            # print(x.shape)
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, rot_emb = frame_pos_emb) + x
            # print(x.shape)
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = fp, rot_emb = image_pos_emb) + x
            # print(x.shape)
            x = ff(x) + x
            # print(x.shape)

        x = self.final_fc(x)

        # print(x.shape)

        return x

class TimeSformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        num_frames = 16,
        image_size = 128,
        patch_size = 8,
        channels = 2,
        depth = 8,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True,
        shift_tokens = False,
        nclass=18,
        label_conc=10,
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions, dim * 2)
        self.label_embedding = nn.Embedding(nclass, num_positions)

        # dim (1024) + label_conc (1024) > 2048
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout, label_conc = label_conc)
            time_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, label_conc = label_conc)
            spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, label_conc = label_conc)

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim + label_conc, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

    def forward(self, x, cond, lc, frame_mask = None):
        device = x.device
        f, hp, wp = x.size(2), x.size(3), x.size(4)
        n = hp * wp

        x = rearrange(x, 'b c f h w -> b (f h w) c')

        cond = self.label_embedding(cond)
        cond = repeat(cond, 'm n -> m n k', k=lc)
        x = torch.cat([x, cond], 2)

        # positional embedding
        frame_pos_emb = None
        image_pos_emb = None

        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, rot_emb = frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, rot_emb = image_pos_emb) + x
            x = ff(x) + x

        return x