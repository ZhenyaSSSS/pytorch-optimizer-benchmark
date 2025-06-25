import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_dim=256, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, dim, H', W')
        x = x.flatten(2)  # (B, dim, N)
        x = x.transpose(1, 2)  # (B, N, dim)
        return x


class MixerLayer(nn.Module):
    def __init__(self, n_patches, emb_dim, token_dim, channel_dim, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, n_patches),
            nn.Dropout(drop),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(emb_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, emb_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):  # (B, N, C)
        # Token-mixing part
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, C, N)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # (B, N, C)
        x = x + y

        # Channel-mixing part
        y = self.norm2(x)
        x = x + self.channel_mlp(y)
        return x


class SmallMLPMixer(nn.Module):
    """MLP-Mixer suitable for CIFAR-10 (≈ 0.7 M params)."""

    def __init__(self, img_size=32, patch_size=4, emb_dim=256, depth=8, num_classes=10, token_dim=128, channel_dim=1024):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, patch_size, emb_dim, img_size)
        n_patches = self.patch_embed.n_patches
        self.mixer_layers = nn.Sequential(*[
            MixerLayer(n_patches, emb_dim, token_dim, channel_dim) for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, N, C)
        x = self.mixer_layers(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # global average pooling over tokens
        return self.head(x) 