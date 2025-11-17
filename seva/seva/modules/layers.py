import math

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from typing import List, Dict, Tuple, Optional

from .transformer import MultiviewTransformer


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> torch.Tensor:
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class Upsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.float()).type(input.dtype)


class TimestepEmbedSequential(nn.Sequential):
    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
        dense_emb: torch.Tensor,
        num_frames: int,
        run_mask: Optional[torch.Tensor] = None,
        run_mask_cfg: Optional[torch.Tensor] = None,
        sparse_mask: Optional[list[torch.Tensor]] = None,
        block_mask: Optional[torch.Tensor] = None,
        meta:Optional[list] = None,
        cache_latents:  Optional[list[Optional[torch.Tensor]]] = None,  
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, MultiviewTransformer):
                # print("layer:", layer.__class__.__name__)
                
                assert num_frames is not None
                x, cache_latents = layer(x, context, num_frames, run_mask,run_mask_cfg,sparse_mask, cache_latents)
            elif isinstance(layer, ResBlock):
                # print("layer:", layer.__class__.__name__)
                
                x = layer(x, emb, dense_emb, block_mask, meta)
            else:
                # print("layer:", layer)
                x = layer(x)
        return x, cache_latents


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None,
        dense_in_channels: int,
        dropout: float,
    ):
        super().__init__()
        out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, out_channels)
        )
        self.dense_emb_layers = nn.Sequential(
            nn.Conv2d(dense_in_channels, 2 * channels, 1, 1, 0)
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1, 1, 0)
            
            
            
    @staticmethod        
    def pack_sparse_blocks_with_halo_unified(
        x: torch.Tensor,              # (B,C,H,W)
        block_mask: torch.Tensor,     # (Hb,Wb) or (1,Hb,Wb) or (B,Hb,Wb), but unified across batch
        meta: Dict[str, int],         # contains H,W,Hb,Wb,block (pad_h/pad_w unused here)
    ) -> Tuple[List[torch.Tensor], Dict[str, object]]:
        # ---- DO NOT CHANGE (as requested) ----
        # print("x shape:", x.shape)
        B, C, H, W = x.shape
        b     = meta["block"]
        Hb    = meta["Hb"];  Wb    = meta["Wb"]
        # pad_h = meta["pad_h"]; pad_w = meta["pad_w"]
        ks = b + 2  # halo = 1 by construction

        cols = F.unfold(x, kernel_size=ks, stride=b, padding=1)  # (B, C*ks*ks, Hb*Wb)
        # --------------------------------------
        # print("cols shape:", cols.shape)

        # Normalize unified mask to (Hb*Wb,) indices ONCE
        bm = block_mask[0]
        # print("bm shape:", bm.shape)
        # assert bm.shape == (Hb, Wb), f"block_mask must be (Hb,Wb); got {tuple(bm.shape)}"
        idx = bm.reshape(-1).nonzero(as_tuple=False).squeeze(1)    # (K,)

        # Select the same K tiles for every batch item and reshape to patches
        sel = cols[:, :, idx]     
        K = sel.shape[-1]
        patches = sel.permute(0, 2, 1).reshape(B * K, C, ks, ks).contiguous()

        # patches_per_batch: List[torch.Tensor] = [patches[bi] for bi in range(B)]  # each (K,C,ks,ks)

        pack_meta = {
            **meta,
            "idx": idx,                 # unified tile indices (K,)
            "idx_list": [idx] * B,      # for APIs expecting per-batch lists
            "C_in": C,
            "B": B,
            "halo": 1,                  # because ks = b + 2
        }
        return patches, pack_meta, K
    
    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, dense_emb: torch.Tensor, block_mask: Optional[torch.Tensor] = None, meta:Optional[list] = None,
    ) -> torch.Tensor:
        in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        B, _, H, W = x.shape
        if block_mask is not None and H == 72:
            # x_cache = x
            b = meta["block"]
            h, pack_meta, K = self.pack_sparse_blocks_with_halo_unified(x, block_mask, meta)
            # print("h size:", h.shape)
            h = in_rest(h)
            # print("dense emb shape:", dense_emb.shape)
            dense = self.dense_emb_layers(
                F.interpolate(
                    dense_emb, size=h.shape[2:], mode="bilinear", align_corners=True
                )
            ).type(h.dtype)
            # print("dense shape:", dense.shape)
            dense_scale, dense_shift = torch.chunk(dense, 2, dim=1)
            dense_scale = torch.repeat_interleave(dense_scale, repeats=K, dim=0)  # (B*K, C, H, W)
            dense_shift = torch.repeat_interleave(dense_shift, repeats=K, dim=0)  # (B*K, C, H, W)
            h = h * (1 + dense_scale) + dense_shift
            h = F.conv2d(
                    h, in_conv.weight, in_conv.bias,
                    stride=in_conv.stride, padding=0
                )        
            emb_out = self.emb_layers(emb).type(h.dtype)
            # print("emb out shape:", emb_out.shape)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            emb_out = torch.repeat_interleave(emb_out, repeats=K, dim=0)  # (B*K, C)
            # print("h shape:", h.shape)
            C = h.shape[1]
            h = h + emb_out
            h = self.out_layers(h)
            h_bk = h.view(B, K, C, b, b).permute(0, 2, 3, 4, 1) 
            h_bk = h_bk.reshape(B, C * b * b, K) 
            # x_cache = F.unfold(x_cache, kernel_size=b, stride=b, padding=0)
            cols = h.new_zeros(B, C * b * b, meta["Hb"] * meta["Wb"]) 
            cols[:,:,pack_meta['idx']] = h_bk
            cols = F.fold(
                cols,                    # (B, C*b*b, Hb*Wb)
                output_size=(H, W),
                kernel_size=b,
                stride=b,
                padding=0                  # must match the unfold padding
            )
            # print("cols shape:", cols.shape)
            h = self.skip_connection(x) + cols
        else:
            h = in_rest(x)
            # print("dense emb shape:", dense_emb.shape)
            dense = self.dense_emb_layers(
                F.interpolate(
                    dense_emb, size=h.shape[2:], mode="bilinear", align_corners=True
                )
            ).type(h.dtype)
            # print("dense shape:", dense.shape)
            dense_scale, dense_shift = torch.chunk(dense, 2, dim=1)
            h = h * (1 + dense_scale) + dense_shift
            h = in_conv(h)
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            h = h + emb_out
            h = self.out_layers(h)
            h = self.skip_connection(x) + h
        return h
