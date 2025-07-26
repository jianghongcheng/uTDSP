import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1):
        super(AttentionModule, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Q, K, V projections
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3, bias=True)
        # Output projection
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_channels)
        # Dropout (optional)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, in_channels)
        B, C = x.shape
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Compute Q, K, V projections and split heads
        qkv = self.qkv_proj(x)  # Shape: (batch_size, in_channels * 3)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)  # Shape: (3, batch_size, num_heads, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Shapes: (batch_size, num_heads, head_dim)
        
        # Compute scaled dot-product attention
        attn_scores = torch.einsum('bhd,bjd->bhj', Q, K) * self.scale  # Shape: (batch_size, num_heads, num_heads)
        attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: (batch_size, num_heads, num_heads)
        
        # Compute attention output
        attn_output = torch.einsum('bhj,bjd->bhd', attn_weights, V)  # Shape: (batch_size, num_heads, head_dim)
        attn_output = attn_output.contiguous().view(B, C)  # Shape: (batch_size, in_channels)
        
        # Output projection
        out = self.fc(attn_output)  # Shape: (batch_size, out_channels)
        
        # Dropout
        out = self.dropout(out)
        
        return out


def timestep_embedding(timesteps, dim, max_period=10000): # 10000
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


@dataclass
class NetConfig:
    """
    default Transformer
    """
    num_channels: int
    skip_layers: Tuple[int]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int = 64
    activation: Activation = Activation.silu
    use_norm: bool = True
    condition_bias: float = 1
    dropout: float = 0
    last_act: Activation = Activation.none
    num_time_layers: int = 2
    time_last_act: bool = False

    def make_model(self):
        return SkipNet(self)







class LNAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: bool,
            use_cond: bool,
            activation: Activation,
            cond_channels: int,
            condition_bias: float = 0,
            dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.cond_layers = AttentionModule(cond_channels, out_channels)

        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None): # [512, 128]
        x = self.linear(x) # [512, 512]

        if self.use_cond:
            cond = self.cond_layers(cond)

            x = self.norm(x) # [512, 512]
        else:
            # no condition
            x = self.norm(x)
        
        x = self.act(x)
        x = self.dropout(x) # ([512, 512]
        return x


class SkipNet(nn.Module):
    """
    concat x to hidden layers
    """
    def __init__(self, conf: NetConfig):
        super().__init__()
        self.conf = conf

        layers = []
        for i in range(conf.num_time_layers):
            # print(f" i {i}")
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_channels
            else:
                a = conf.num_channels
                b = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                cond = False
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

            self.layers.append(
                LNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=conf.num_channels,
                    use_cond=cond,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                ))
        self.last_act = conf.last_act.get_act()

    def forward(self, x, t, **kwargs):
        t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x, cond=cond)
        x = self.last_act(x)

        return x
