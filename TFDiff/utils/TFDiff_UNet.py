# Code from DiffTraj model: https://github.com/Yasoz/DiffTraj

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F


def create_timestep_embedding(time_steps, embed_dim):
    """
    Create sinusoidal timestep embeddings.
    time_steps: 1D tensor of time indices (shape: [batch_size])
    embed_dim:  size of the embedding
    """
    assert len(time_steps.shape) == 1

    half_dimension = embed_dim // 2
    base = np.log(10000) / (half_dimension - 1)
    scale = torch.exp(torch.arange(half_dimension, dtype=torch.float32) * -base)
    scale = scale.to(device=time_steps.device)
    scaled_time = time_steps.float()[:, None] * scale[None, :]
    embedding_vector = torch.cat([torch.sin(scaled_time),
                                  torch.cos(scaled_time)], dim=1)
    # Zero-pad if embedding dimension is odd
    if embed_dim % 2 == 1:
        embedding_vector = F.pad(embedding_vector, (0, 1, 0, 0))
    return embedding_vector


class AttentionLayer(nn.Module):
    """
    Applies a simple attention mechanism over feature embeddings:
       scores = softmax(Linear(features))
    """
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embed_dim)
        attention_scores = self.attention_fc(x)  # (batch_size, num_attributes, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights

class WideDeepNetwork(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256):
        """
        Wide part:  linear layer for continuous attributes
        Deep part:  embedding + MLP for categorical attributes
        """
        super(WideDeepNetwork, self).__init__()

        # Wide part
        self.linear_wide = nn.Linear(1, embed_dim)

        # Deep part (for categorical attributes)
        self.time_emb = nn.Embedding(48, hidden_dim)
        self.day_emb = nn.Embedding(7, hidden_dim)
        self.max_id_emb = nn.Embedding(100, hidden_dim)

        self.fc1_deep = nn.Linear(hidden_dim * 3, embed_dim)
        self.fc2_deep = nn.Linear(embed_dim, embed_dim)

    def forward(self, attributes):
        # Continuous attribute (e.g. mean or max traffic)
        continuous_features = attributes[:, 2:3]

        # Categorical attributes
        t_idx = attributes[:, 0].long()     # time
        d_idx = attributes[:, 1].long()     # day
        midx = attributes[:, 3].long()      # max_id

        # Wide output
        wide_output = self.linear_wide(continuous_features)

        # Deep output
        time_embedding = self.time_emb(t_idx)
        day_embedding = self.day_emb(d_idx)
        max_id_embedding = self.max_id_emb(midx)

        cat_embed = torch.cat([time_embedding, day_embedding, max_id_embedding], dim=1)
        deep_output = F.relu(self.fc1_deep(cat_embed))
        deep_output = self.fc2_deep(deep_output)

        # Combine
        combined_output = deep_output + wide_output
        return combined_output


def sigmoid_activation(x):
    return x * torch.sigmoid(x)


def GroupNorm32(in_channels):
    """
    Applies Group Normalization with 32 groups (and affine parameters).
    """
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels,
                                  kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (1, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None,
                 conv_shortcut=False, dropout=0.1, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = GroupNorm32(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, out_channels,
                                               kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels,
                                              kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = sigmoid_activation(h)
        h = self.conv1(h)
        h += self.temb_proj(sigmoid_activation(temb))[:, :, None]
        h = self.norm2(h)
        h = sigmoid_activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        normed = self.norm(x)
        q_data = self.q(normed)
        k_data = self.k(normed)
        v_data = self.v(normed)

        b, c, width = q_data.shape
        q_data = q_data.permute(0, 2, 1)        # (b, width, c)
        w_ = torch.bmm(q_data, k_data)          # (b, width, width)
        w_ *= c**(-0.5)
        w_ = F.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)                # (b, width, width)

        attn = torch.bmm(v_data, w_)
        attn = attn.reshape(b, c, width)
        attn = self.proj_out(attn)
        return x + attn


class UNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ch = config.model.ch
        out_ch = config.model.out_ch
        ch_mult = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.resolution
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # Input convolution
        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # Downsampling modules
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in_channels = None

        for i_level in range(self.num_resolutions):
            blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()

            block_in_channels = ch * in_ch_mult[i_level]
            block_out_channels = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        in_channels=block_in_channels,
                        out_channels=block_out_channels,
                        temb_channels=self.temb_ch,
                        dropout=dropout
                    )
                )
                block_in_channels = block_out_channels

                if curr_res in attn_resolutions:
                    attn_blocks.append(SelfAttentionBlock(block_in_channels))

            down = nn.Module()
            down.block = blocks
            down.attn = attn_blocks

            if i_level != self.num_resolutions - 1:
                down.downsample = DownSampleBlock(block_in_channels, resamp_with_conv)
                curr_res //= 2

            self.down.append(down)

        # Middle layer
        self.mid = nn.Module()
        self.mid.block_1 = ResidualBlock(block_in_channels, block_in_channels,
                                         temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = SelfAttentionBlock(block_in_channels)
        self.mid.block_2 = ResidualBlock(block_in_channels, block_in_channels,
                                         temb_channels=self.temb_ch, dropout=dropout)

        # Upsampling modules
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()

            block_out_channels = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]

                blocks.append(
                    ResidualBlock(
                        in_channels=block_in_channels + skip_in,
                        out_channels=block_out_channels,
                        temb_channels=self.temb_ch,
                        dropout=dropout
                    )
                )
                block_in_channels = block_out_channels

                if curr_res in attn_resolutions:
                    attn_blocks.append(SelfAttentionBlock(block_in_channels))

            up = nn.Module()
            up.block = blocks
            up.attn = attn_blocks

            if i_level != 0:
                up.upsample = UpSampleBlock(block_in_channels, resamp_with_conv)
                curr_res *= 2

            # Prepend so that the final order matches the forward pass
            self.up.insert(0, up)

        # Final normalization and output convolution
        self.norm_out = GroupNorm32(block_in_channels)
        self.conv_out = nn.Conv1d(block_in_channels, out_ch,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, extra_embed=None):
        # Ensure input resolution matches config
        assert x.shape[2] == self.resolution

        # Timestep embeddings
        temb = create_timestep_embedding(t, self.ch)
        temb = sigmoid_activation(self.temb.dense[0](temb))
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # Downsample
        h_list = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h_list[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h_list.append(h)

            if i_level != self.num_resolutions - 1:
                h_list.append(self.down[i_level].downsample(h_list[-1]))

        # Middle
        h = h_list[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsample
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                skip_tensor = h_list.pop()
                # If needed, pad so widths match
                if skip_tensor.size(-1) != h.size(-1):
                    h = F.pad(h, (0, skip_tensor.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](
                    torch.cat([h, skip_tensor], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Final output
        h = self.norm_out(h)
        h = sigmoid_activation(h)
        h = self.conv_out(h)
        return h

class Guide_UNet(nn.Module):
    def __init__(self, config):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale

        self.unet = UNetModel(config)
        self.guide_emb = WideDeepNetwork(self.ch)
        self.place_emb = WideDeepNetwork(self.ch)

    def forward(self, x, t, attr):
        # Embeddings for conditioning (attr) and "unconditional" (zeros)
        guide_embedding = self.guide_emb(attr)
        zero_attr = torch.zeros_like(attr)
        place_embedding = self.place_emb(zero_attr)

        # Conditional and unconditional predictions
        cond_out = self.unet(x, t, guide_embedding)
        uncond_out = self.unet(x, t, place_embedding)

        # CFG scaling
        final_pred = cond_out + self.guidance_scale * (cond_out - uncond_out)
        return final_pred

if __name__ == '__main__':
    # Example usage
    from utils.config_WD import args

    # Convert dict of dicts to nested SimpleNamespace
    temp_args = {}
    for key, val in args.items():
        temp_args[key] = SimpleNamespace(**val)
    config = SimpleNamespace(**temp_args)

    # Mock time steps
    t = torch.randn(10)

    # Example attributes
    time_day = torch.zeros(10)        # 48 categories
    day = torch.zeros(10)            # 7 categories
    max_traffic = torch.zeros(10)    # continuous
    max_id = torch.zeros(10)         # 100 categories

    # Combine into [time, day, max_traffic, max_id]
    attr = torch.stack([time_day, day, max_traffic, max_id], dim=1)

    # Create guided U-Net
    guided_unet = Guide_UNet(config)
    x = torch.randn(10, 10, 10)

    # Forward pass
    total_params = sum(p.numel() for p in guided_unet.parameters())
    print(f'{total_params:,} total parameters.')

    model_output = guided_unet(x, t, attr)
    print(model_output.shape)
