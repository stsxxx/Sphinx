import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )
        # print("q shape:", q.shape)
        # print("k shape:", k.shape)
        # print("v shape:", v.shape)
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, x_cache: Optional[torch.Tensor] = None, x_cache_flat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # partial
        if x_cache_flat is not None:
            x = self.attn1(self.norm1(x), context=self.norm1(x_cache_flat)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
        elif x_cache is not None:
            x = self.attn1(self.norm1(x), context=self.norm1(x_cache)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
        else:
            x = self.attn1(self.norm1(x)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
        return x


class TransformerBlockTimeMix(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout)
        self.attn1 = Attention(
            query_dim=inner_dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=inner_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, num_frames: int, x_cache: Optional[torch.Tensor] = None, run_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # partial
        if x_cache is not None:
            # _, s, _ = x_cache.shape
            # x = rearrange(x_cache, "(b t) s c -> (b s) t c", t=21)
            # x = self.ff_in(self.norm_in(x)) + x
            # x = self.attn1(self.norm1(x[:,run_mask,:]), context=self.norm1(x)) + x[:,run_mask,:]
            # x = self.attn2(self.norm2(x), context=context) + x
            # x = self.ff(self.norm3(x))
            # x = rearrange(x, "(b s) t c -> (b t) s c", s=s)
            _, s, _ = x_cache.shape
            x = rearrange(x_cache, "(b t) s c -> (b s) t c", t=21)
            x = self.ff_in(self.norm_in(x)) + x
            x = self.attn1(self.norm1(x[:,run_mask,:]), context=self.norm1(x)) + x[:,run_mask,:]
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x))
            x = rearrange(x, "(b s) t c -> (b t) s c", s=s)
        # full
        else:
            _, s, _ = x.shape
            x = rearrange(x, "(b t) s c -> (b s) t c", t=num_frames)
            x = self.ff_in(self.norm_in(x)) + x
            x = self.attn1(self.norm1(x), context=None) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x))
            x = rearrange(x, "(b s) t c -> (b t) s c", s=s)            
        return x


class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x_spatial: torch.Tensor, x_temporal: torch.Tensor
    ) -> torch.Tensor:
        return x_spatial + x_temporal


class MultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.unflatten_names = unflatten_names

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.time_mixer = SkipConnect()
        self.time_mix_blocks = nn.ModuleList(
            [
                TransformerBlockTimeMix(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, num_frames: int, run_mask: Optional[torch.Tensor] = None,
        run_mask_cfg: Optional[torch.Tensor] = None,
        sparse_mask: Optional[list[torch.Tensor]] = None,
        cache_latents:  Optional[list[Optional[torch.Tensor]]] = None,  
    ) -> torch.Tensor:
        assert context.ndim == 3
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        
        if sparse_mask is not None:
            if h == 72:
                sparse_mask = sparse_mask[0]
            elif h == 36:
                sparse_mask = sparse_mask[1]
            elif h == 18:
                sparse_mask = sparse_mask[2]
            elif h == 9:
                sparse_mask = sparse_mask[3]
            else:
                raise ValueError(f"Unsupported spatial size: {h}")
            x = x[:,:,sparse_mask]
            x = rearrange(x, "b c n -> b n c")
            b, n, c = x.shape
            
            time_context = context
            time_context_first_timestep = time_context[::num_frames]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=n
            )
        else:
            time_context = context
            time_context_first_timestep = time_context[::num_frames]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
            x = rearrange(x, "b c h w -> b (h w) c")
            
        # print("name:", self.name)
        # print("unflatten names:", self.unflatten_names)
        if self.name in self.unflatten_names:
            context = context[::num_frames]
            
        x = self.proj_in(x)
        # print("xxx shape:", x.shape)
        # print("context shape:", context.shape)
        # print("time_context shape:", time_context.shape)
        
        # full 
        if run_mask is None:
            for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
                if cache_latents is not None:
                    cache_latents.append(x)
                if self.name in self.unflatten_names:
                    x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w)
                # print("xxxx shape:", x.shape)
                    
                x = block(x, context=context)

                if self.name in self.unflatten_names:
                    x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w)
                if cache_latents is not None:
                    cache_latents.append(x)
                # cache_latents.append(x)
                x_mix = mix_block(x, context=time_context, num_frames=num_frames)
                x = self.time_mixer(x_spatial=x, x_temporal=x_mix)
                x = self.proj_out(x)
                cache_latents.append(x)
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
                
        else:

            
            if sparse_mask is not None:
                x_cache_sparse = None
                x_cache_flat = None
                sparse_mask = sparse_mask.view(-1)
                # print("sparse_mask:", sparse_mask)
                # print("cache_latents shapes1:", [t.shape if t is not None else None for t in cache_latents])
                for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
                    x_cache_sparse = cache_latents.pop(0)
                    # print("x shape:", x.shape)
                    # print("x_cache_sparse shape:", x_cache_sparse[run_mask_cfg, sparse_mask, :].shape)
                    tmp = x_cache_sparse[run_mask_cfg]
                    tmp[:, sparse_mask, :] = x 
                    x_cache_sparse[run_mask_cfg] = tmp
                    if self.name in self.unflatten_names:
                        x = rearrange(x, "(b t) n c -> b (t n) c", t=num_frames, n=n)
                        x_cache_flat = rearrange(x_cache_sparse, "(b t) (h w) c -> b (t h w) c", t=21, h=h, w=w)
                        # print("xxxx shape:", x.shape)
                        # print("xxxx cache shape:", x_cache.shape)
                        
                        
                    x = block(x,  context=context, x_cache_flat=x_cache_flat, x_cache=x_cache_sparse[run_mask_cfg])
                    if self.name in self.unflatten_names:
                        x = rearrange(x, "b (t n) c -> (b t) n c ", t=num_frames, n=n)
                    x_cache_sparse = cache_latents.pop(0)
                    tmp = x_cache_sparse[run_mask_cfg]
                    tmp[:, sparse_mask, :] = x 
                    x_cache_sparse[run_mask_cfg] = tmp
                    
                    x_mix = mix_block(x, context=time_context, num_frames=num_frames, x_cache=x_cache_sparse[:,sparse_mask,:], run_mask=run_mask)
                    x = self.time_mixer(x_spatial=x, x_temporal=x_mix)
                    x = self.proj_out(x)
                    
                    x_cache_sparse = cache_latents.pop(0)
                    
                    tmp = x_cache_sparse[run_mask_cfg]
                    tmp[:, sparse_mask, :] = x 

                    
                    x = rearrange(tmp, "b (h w) c -> b c h w", h=h, w=w)
            else:
                x_cache = None
                for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
                    x_cache_p = cache_latents.pop(0)
                    if self.name in self.unflatten_names:
                        x_cache_p[run_mask_cfg] = x
                        x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w)
                        x_cache = rearrange(x_cache_p, "(b t) (h w) c -> b (t h w) c", t=21, h=h, w=w)
                        # print("xxxx shape:", x.shape)
                        # print("xxxx cache shape:", x_cache.shape)
                        
                        
                    x = block(x,  context=context, x_cache=x_cache)
                    if self.name in self.unflatten_names:
                        x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w)
                    x_cache = cache_latents.pop(0)
                    x_cache[run_mask_cfg] = x
                    x_mix = mix_block(x, context=time_context, num_frames=num_frames, x_cache=x_cache, run_mask=run_mask)
                    x = self.time_mixer(x_spatial=x, x_temporal=x_mix)
                    x = self.proj_out(x)
                    x_cache = cache_latents.pop(0)
                    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        
        out = x + x_in
        return out, cache_latents
