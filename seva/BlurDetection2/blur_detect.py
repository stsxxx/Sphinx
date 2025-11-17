# --- paste into process.py ---

import torch
import torch.nn.functional as F

def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 3, H, W] or [..., 1, H, W] in [0,1]
    returns: [..., 1, H, W]
    """
    if x.shape[-3] == 1:
        return x
    # Rec.709 luma
    r = x[..., -3, :, :]
    g = x[..., -2, :, :]
    b = x[..., -1, :, :]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y.unsqueeze(-3)

def _laplacian_5pt(img1: torch.Tensor) -> torch.Tensor:
    """
    img1: [..., 1, H, W], float
    returns: Laplacian response [..., 1, H, W]
    """
    device = img1.device
    lap = torch.tensor([[0.,  1., 0.],
                        [1., -4., 1.],
                        [0.,  1., 0.]], device=device, dtype=img1.dtype)
    k = lap.view(1, 1, 3, 3)
    return F.conv2d(img1, k, padding=1)

def _gaussian_kernel(ks: int = 5, sigma: float = 1.0, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(ks, device=device, dtype=dtype) - (ks - 1)/2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel2d = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ks,ks]
    return kernel2d

def _smooth(img1: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Separable Gaussian via 2D kernel (single conv) for simplicity.
    img1: [..., 1, H, W]
    """
    if sigma <= 0:
        return img1
    ks = max(3, int(2 * round(3 * sigma) + 1))  # ~6σ rule, odd
    k = _gaussian_kernel(ks, sigma, device=img1.device, dtype=img1.dtype)
    return F.conv2d(img1, k, padding=ks // 2)

def _normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample min-max normalize to [0,1]
    x: [B,1,H,W]
    """
    x_min = x.amin(dim=(-1, -2), keepdim=True)
    x_max = x.amax(dim=(-1, -2), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _otsu_threshold_per_batch(x01: torch.Tensor, nbins: int = 256) -> torch.Tensor:
    """
    x01: [B,1,H,W] in [0,1]
    returns: thresholds [B,1,1,1] in [0,1]
    """
    B = x01.shape[0]
    flat = x01.view(B, -1)
    # Histogram per batch
    # (Use linspace bin edges to be consistent)
    bins = torch.linspace(0, 1, steps=nbins+1, device=x01.device, dtype=x01.dtype)
    hist = []
    for b in range(B):
        h = torch.histc(flat[b].float(), bins=nbins, min=0.0, max=1.0).to(x01.dtype)
        hist.append(h)
    hist = torch.stack(hist, dim=0)  # [B, nbins]
    p = hist / (hist.sum(dim=1, keepdim=True) + 1e-12)

    # Cumulative sums
    omega = torch.cumsum(p, dim=1)                    # class prob up to t
    mu_k = torch.cumsum(p * torch.linspace(0, 1, nbins, device=x01.device, dtype=x01.dtype), dim=1)
    mu_T = mu_k[:, -1:]                               # total mean

    # Between-class variance: sigma_b^2 = (mu_T*omega - mu_k)^2 / (omega*(1-omega))
    num = (mu_T * omega - mu_k)**2
    den = omega * (1 - omega) + 1e-12
    sigma_b2 = num / den

    # Best threshold index
    t_idx = torch.argmax(sigma_b2, dim=1)            # [B]
    # Map index -> threshold in [0,1]
    thr = (t_idx.float() / (nbins - 1)).view(B, 1, 1, 1)
    return thr

def blur_mask_from_rendered_torch(
    rendered: torch.Tensor,
    method: str = "otsu",          # "otsu" | "percentile" | "mad" | "fixed"
    percentile: float = 70.0,      # used if method="percentile"
    mad_k: float = 1.5,            # used if method="mad"
    fixed_thr: float = 0.25,       # in [0,1], used if method="fixed" (on normalized |Lap|)
    smooth_sigma: float = 1.0,     # Gaussian sigma before thresholding
    clean: bool = True,            # simple morphology cleanups
    morph_k: int = 3
) -> torch.Tensor:
    """
    rendered: [V,3,H,W] or [3,H,W] or [H,W] torch.float (0..1)
    returns:  mask01: [V,1,H,W] (or [1,H,W]) float in {0,1}, where 1=BLUR (low Laplacian magnitude)
    """
    x = rendered
    added_batch = False
    if x.dim() == 2:         # [H,W] -> [1,1,H,W]
        x = x.unsqueeze(0).unsqueeze(0)
        added_batch = True
    elif x.dim() == 3:       # [C,H,W] -> [1,C,H,W]
        x = x.unsqueeze(0)
        added_batch = True
    # now x is [B,C,H,W]
    x = _to_gray(x)          # [B,1,H,W]

    # Laplacian magnitude
    lap = _laplacian_5pt(x)          # [B,1,H,W]
    mag = lap.abs()
    mag = _smooth(mag, smooth_sigma) # optional smoothing
    mag01 = _normalize_01(mag)       # normalize per-sample to [0,1]

    # Blur = low magnitude ⇒ threshold on mag01
    if method == "otsu":
        thr = _otsu_threshold_per_batch(mag01)                   # [B,1,1,1]
        mask01 = (mag01 <= thr).to(mag01.dtype)
    elif method == "percentile":
        q = torch.tensor(percentile/100.0, device=mag01.device, dtype=mag01.dtype)
        flat = mag01.view(mag01.shape[0], -1)
        thr_vals = torch.quantile(flat, q, dim=1, interpolation="nearest").view(-1,1,1,1)
        mask01 = (mag01 <= thr_vals).to(mag01.dtype)
    elif method == "mad":
        flat = mag01.view(mag01.shape[0], -1)
        med  = flat.median(dim=1).values.view(-1,1,1,1)
        mad  = (flat - med.view(-1,1)).abs().median(dim=1).values.view(-1,1,1,1) + 1e-8
        thr  = med - mad_k * 1.4826 * mad
        mask01 = (mag01 <= thr).to(mag01.dtype)
    elif method == "fixed":
        mask01 = (mag01 <= fixed_thr).to(mag01.dtype)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Optional: light cleanup (close then open) using pooling, stays on GPU
    if clean:
        # dilation: any 1 in k×k window
        dil = (F.max_pool2d(mask01, kernel_size=morph_k, stride=1, padding=morph_k//2) > 0).to(mask01.dtype)
        # erosion: all 1s in k×k window (duality)
        ero = 1.0 - (F.max_pool2d(1.0 - dil, kernel_size=morph_k, stride=1, padding=morph_k//2) > 0).to(mask01.dtype)
        # open then close for speckle removal + hole fill
        # open
        ero2 = 1.0 - (F.max_pool2d(1.0 - mask01, kernel_size=morph_k, stride=1, padding=morph_k//2) > 0).to(mask01.dtype)
        open_ = (F.max_pool2d(ero2, kernel_size=morph_k, stride=1, padding=morph_k//2) > 0).to(mask01.dtype)
        # close
        dil2 = (F.max_pool2d(open_, kernel_size=morph_k, stride=1, padding=morph_k//2) > 0).to(mask01.dtype)
        mask01 = dil2

    if added_batch:
        mask01 = mask01.squeeze(0)   # back to [1,H,W] or [H,W] if you prefer
    return mask01
