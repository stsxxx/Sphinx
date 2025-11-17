import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from typing import Sequence
from seva.geometry import get_camera_dist
from typing import Optional

def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def make_betas(
    num_timesteps: int, linear_start: float = 1e-4, linear_end: float = 2e-2
) -> np.ndarray:
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64
        )
        ** 2
    )
    return betas.numpy()


def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class EpsScaling(object):
    def __call__(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class DDPMDiscretization(object):
    def __init__(
        self,
        linear_start: float = 5e-06,
        linear_end: float = 0.012,
        num_timesteps: int = 1000,
        log_snr_shift: float | None = 2.4,
    ):
        self.num_timesteps = num_timesteps

        betas = make_betas(
            num_timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        self.log_snr_shift = log_snr_shift

        alphas = 1.0 - betas  # first alpha here is on data side
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(f"Expected n <= {self.num_timesteps}, but got n = {n}.")

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        if self.log_snr_shift is not None:
            sigmas = sigmas * np.exp(self.log_snr_shift)
        return torch.flip(
            torch.tensor(sigmas, dtype=torch.float32, device=device), (0,)
        )

    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        flip: bool = False,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))


class DiscreteDenoiser(object):
    sigmas: torch.Tensor

    def __init__(
        self,
        discretization: DDPMDiscretization,
        num_idx: int = 1000,
        device: str | torch.device = "cpu",
    ):
        self.scaling = EpsScaling()
        self.discretization = discretization
        self.num_idx = num_idx
        self.device = device

        self.register_sigmas()

    def register_sigmas(self):
        self.sigmas = self.discretization(
            self.num_idx, do_append_zero=False, flip=True, device=self.device
        )

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: torch.Tensor | int) -> torch.Tensor:
        return self.sigmas[idx]

    def __call__(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: dict,
        run_mask: Optional[torch.Tensor] = None,
        sparse_mask_: Optional[list[torch.Tensor]] = None,
        block_mask_: Optional[torch.Tensor] = None,
        meta_:Optional[list] = None,
        cache_latents:  Optional[list[Optional[torch.Tensor]]] = None,  
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.idx_to_sigma(self.sigma_to_idx(sigma))
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.sigma_to_idx(c_noise.reshape(sigma_shape))
        # print(f"c para: {c_skip}, {c_out}, {c_in}, {c_noise}")
        # print("cond:", cond)
        # if "replace" in cond and run_mask is None:
        if "replace" in cond:
        
            x, mask = cond.pop("replace").split((input.shape[1], 1), dim=1)
            input = input * (1 - mask) + x * mask
            # print("input shape:", input.shape)
            # print("mask shape:", mask.shape)
            
            # print("mask:", mask[:,:,0,0])
        out, updated_cache_latents = network(
            input * c_in, c_noise, cond, run_mask, sparse_mask_, block_mask_, meta_, cache_latents, **additional_model_inputs
        )

        return out * c_out + input * c_skip, updated_cache_latents
        # return (
        #     network(input * c_in, c_noise, cond, run_mask, cache_latents, **additional_model_inputs) * c_out
        #     + input * c_skip
        # )


class ConstantScaleRule(object):
    def __call__(self, scale: float | torch.Tensor) -> float | torch.Tensor:
        return scale


class MultiviewScaleRule(object):
    def __init__(self, min_scale: float = 1.0):
        self.min_scale = min_scale

    def __call__(
        self,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        c2w_input = c2w[input_frame_mask]
        rotation_diff = get_camera_dist(c2w, c2w_input, mode="rotation").min(-1).values
        translation_diff = (
            get_camera_dist(c2w, c2w_input, mode="translation").min(-1).values
        )
        K_diff = (
            ((K[:, None] - K[input_frame_mask][None]).flatten(-2) == 0).all(-1).any(-1)
        )
        close_frame = (rotation_diff < 10.0) & (translation_diff < 1e-5) & K_diff
        if isinstance(scale, torch.Tensor):
            scale = scale.clone()
            scale[close_frame] = self.min_scale
        elif isinstance(scale, float):
            scale = torch.where(close_frame, self.min_scale, scale)
        else:
            raise ValueError(f"Invalid scale type {type(scale)}.")
        return scale


class ConstantScaleSchedule(object):
    def __call__(
        self, sigma: float | torch.Tensor, scale: float | torch.Tensor
    ) -> float | torch.Tensor:
        if isinstance(sigma, float):
            return scale
        elif isinstance(sigma, torch.Tensor):
            if len(sigma.shape) == 1 and isinstance(scale, torch.Tensor):
                sigma = append_dims(sigma, scale.ndim)
            return scale * torch.ones_like(sigma)
        else:
            raise ValueError(f"Invalid sigma type {type(sigma)}.")


class ConstantGuidance(object):
    def __call__(
        self,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        scale: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(scale, torch.Tensor) and len(scale.shape) == 1:
            scale = append_dims(scale, cond.ndim)
        return uncond + scale * (cond - uncond)


class VanillaCFG(object):
    def __init__(self):
        self.scale_rule = ConstantScaleRule()
        self.scale_schedule = ConstantScaleSchedule()
        self.guidance = ConstantGuidance()

    def __call__(
        self, x: torch.Tensor, sigma: float | torch.Tensor, scale: float | torch.Tensor
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        scale = self.scale_rule(scale)
        scale_value = self.scale_schedule(sigma, scale)
        x_pred = self.guidance(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict, run_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        c_out = dict()
        if run_mask is not None:
            for k in c:
                if k in ["vector", "crossattn",  "concat", "replace", "dense_vector"]:
                    c_out[k] = torch.cat((uc[k][run_mask], c[k][run_mask]), 0)
                else:
                    assert c[k] == uc[k]
                    c_out[k] = c[k]
        else:
            for k in c:
                if k in ["vector", "crossattn", "concat", "replace", "dense_vector"]:
                    c_out[k] = torch.cat((uc[k], c[k]), 0)
                else:
                    assert c[k] == uc[k]
                    c_out[k] = c[k]
        # print("c out:", c_out)
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class MultiviewCFG(VanillaCFG):
    def __init__(self, cfg_min: float = 1.0):
        self.scale_min = cfg_min
        self.scale_rule = MultiviewScaleRule(min_scale=cfg_min)
        self.scale_schedule = ConstantScaleSchedule()
        self.guidance = ConstantGuidance()

    def __call__(  # type: ignore
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        scale: float | torch.Tensor,
        run_mask: Optional[torch.Tensor],
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
        
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        # print("c2w:", c2w.shape)
        # print("K:", K.shape)
        # print("input_frame_mask:", input_frame_mask.shape)
        
        # if run_mask is not None:
        #     c2w = c2w[run_mask]
        #     K = K[run_mask]
        #     input_frame_mask = input_frame_mask[run_mask]
        scale = self.scale_rule(scale, c2w, K, input_frame_mask)
        # print("scale:", scale)
        # print("sigma:", sigma)
        # print("run mask:", run_mask)
        if run_mask is not None:
            scale_value = self.scale_schedule(sigma, scale[run_mask])
        else:
            scale_value = self.scale_schedule(sigma, scale)
        x_pred = self.guidance(x_u, x_c, scale_value)
        return x_pred


class MultiviewTemporalCFG(MultiviewCFG):
    def __init__(self, num_frames: int, cfg_min: float = 1.0):
        super().__init__(cfg_min=cfg_min)

        self.num_frames = num_frames
        distance_matrix = (
            torch.arange(num_frames)[None] - torch.arange(num_frames)[:, None]
        ).abs()
        self.distance_matrix = distance_matrix

    def __call__(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_frame_mask = rearrange(
            input_frame_mask, "(b t) ... -> b t ...", t=self.num_frames
        )
        min_distance = (
            self.distance_matrix[None].to(x.device)
            + (~input_frame_mask[:, None]) * self.num_frames
        ).min(-1)[0]
        min_distance = min_distance / min_distance.max(-1, keepdim=True)[0].clamp(min=1)
        scale = min_distance * (scale - self.scale_min) + self.scale_min
        scale = rearrange(scale, "b t ... -> (b t) ...")
        scale = append_dims(scale, x.ndim)
        return super().__call__(x, sigma, scale, c2w, K, input_frame_mask.flatten(0, 1))


class EulerEDMSampler(object):
    def __init__(
        self,
        discretization: DDPMDiscretization,
        guider: VanillaCFG | MultiviewCFG | MultiviewTemporalCFG,
        num_steps: int | None = None,
        verbose: bool = False,
        device: str | torch.device = "cuda",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
        self.num_steps = num_steps
        self.discretization = discretization
        self.guider = guider
        self.verbose = verbose
        self.device = device

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def prepare_sampling_loop(
        self, x: torch.Tensor, cond: dict, uc: dict, num_steps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict, dict]:
        num_steps = num_steps or self.num_steps
        assert num_steps is not None, "num_steps must be specified"
        sigmas = self.discretization(num_steps, device=self.device)
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def get_sigma_gen(self, num_sigmas: int, verbose: bool = True) -> range | tqdm:
        sigma_generator = range(num_sigmas - 1)
        if self.verbose and verbose:
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas - 1,
                desc="Sampling",
                leave=False,
            )
        return sigma_generator

    def sampler_step(
        self,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict,
        gamma: float = 0.0,
        run_mask: Optional[torch.Tensor] = None,
        **guider_kwargs,
    ) -> torch.Tensor:
        # print("cond:", cond)
        # print("uc:", uc)
        # print("sigma:", sigma)
        # print("next_sigma:", next_sigma)
        
        sigma_hat = sigma * (gamma + 1.0) + 1e-6

        eps = torch.randn_like(x) * self.s_noise
        x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised, _ = denoiser(*self.guider.prepare_inputs(x, sigma_hat, cond, uc))
        denoised = self.guider(denoised, sigma_hat, scale, run_mask, **guider_kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        return x + dt * d
    
    def sampler_step_finer(
        self,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict,
        run_mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[list[torch.Tensor]] = None,
        block_mask: Optional[torch.Tensor] = None,
        meta: Optional[dict] = None,
        cache_latents:  Optional[list[Optional[torch.Tensor]]] = None,  
        gamma: float = 0.0,
        **guider_kwargs,
    ) -> torch.Tensor:
        # print("cond:", cond)
        # print("uc:", uc)
        # print("sigma:", sigma)
        # print("next_sigma:", next_sigma)
        
        sigma_hat = sigma * (gamma + 1.0) + 1e-6
        # print("guider:", self.guider)
        eps = torch.randn_like(x) * self.s_noise
        x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised, cache_latents = denoiser(*self.guider.prepare_inputs(x, sigma_hat, cond, uc, run_mask), run_mask, sparse_mask, block_mask, meta, cache_latents)
        # print("denoised shape:", denoised.shape)
        # print("scale:", scale)
        # print("sigma_hat shape:", sigma_hat.shape)
        denoised = self.guider(denoised, sigma_hat, scale, run_mask, **guider_kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        return x + dt * d, cache_latents
    
    
    def __call__(
        self,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict | None = None,
        num_steps: int | None = None,
        verbose: bool = True,
        **guider_kwargs,
    ) -> torch.Tensor:
        uc = cond if uc is None else uc
        # print("x mean init :", x.mean())
        # print("random latent",x.mean().item(), x.std().item())

        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x,
            cond,
            uc,
            num_steps,
        )
        # print("x mean scaled :", x.mean())
        # print("sigmas:",sigmas)
        # print("num of sigmas:", num_sigmas)
        run_mask = None
        for i in self.get_sigma_gen(num_sigmas, verbose=verbose):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                scale,
                cond,
                uc,
                gamma,
                run_mask,
                **guider_kwargs,
            )
            # print("x input:", x[0:2])
        return x


    def partial_denoising(
        self,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict,
        input_masks: torch.Tensor,
        s: int,
        num_steps: int | None = None,
        verbose: bool = True,
        **guider_kwargs,
    ) -> torch.Tensor:
        """
        Add noise to x[~input_masks] to simulate step `s`, then denoise from step `s` to the end.
        Assumes x is scaled latent from Autoencoder (with factor 0.18215).
        """
        uc = cond if uc is None else uc
        num_steps = num_steps or self.num_steps
        assert num_steps is not None, "num_steps must be specified"

        # Get sigma schedule
        sigmas = self.discretization(num_steps, device=self.device)
        num_sigmas = len(sigmas)
        assert 0 <= s < num_sigmas, f"Invalid step s={s}, must be < {num_sigmas}"

        # Add noise to simulate step s
        x_scaled = x.clone()
        noise = torch.randn_like(x_scaled) * self.s_noise
        x_scaled += noise * sigmas[s]

        s_in = x.new_ones([x.shape[0]])
        run_mask = None
        for i in self.get_sigma_gen(num_sigmas, verbose=verbose):
            if i < s:
                continue

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )

            x_scaled = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x_scaled,  # ← no manual scaling
                scale,
                cond,
                uc,
                gamma,
                run_mask,
                **guider_kwargs,
            )

        return x_scaled

    def partial_denoising_finer(
        self,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict,
        input_masks: torch.Tensor,
        s: list,
        sparse_mask: list[torch.Tensor],
        block_mask: torch.Tensor,
        meta: dict,
        num_steps: int | None = None,
        verbose: bool = True,
        **guider_kwargs,
    ) -> torch.Tensor:
        # for k, v in guider_kwargs.items():
        #     print(f"{k} : {v.shape}")
        
        mask_hw = sparse_mask[0].unsqueeze(0).unsqueeze(0) if sparse_mask is not None else None
        
        start_t = min(s[1:-1])
        s = torch.as_tensor(s, dtype=torch.long, device=self.device)

        uc = cond if uc is None else uc
        num_steps = num_steps or self.num_steps
        assert num_steps is not None, "num_steps must be specified"

        # Get sigma schedule
        sigmas = self.discretization(num_steps, device=self.device)
        num_sigmas = len(sigmas)
        assert 0 <= start_t < num_sigmas, f"Invalid step s={start_t}, must be < {num_sigmas}"

        # Add noise to simulate step s
        x_scaled = x.clone()
        x_scaled_copy = x.clone()
        
        noise = torch.randn_like(x_scaled) * self.s_noise
        x_scaled += noise * sigmas[start_t]
        # print("sigmas:", sigmas)
        s_in = x.new_ones([x.shape[0]])
        # print("scale:", scale)
        cache_latents = []
        for i in self.get_sigma_gen(num_sigmas, verbose=verbose):
            if i < start_t:
                continue
            # run_mask = [1 if s_j <= i else 0 for s_j in s]
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            # print("cache latents:", cache_latents)
            # print("cache_latents shapes:", [t.shape if t is not None else None for t in cache_latents])
            
            if i % 5 == 0 or i >= 40:
                run_mask = None
                x_scaled, cache_latents = self.sampler_step_finer(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x_scaled,  # ← no manual scaling
                    scale,
                    cond,
                    uc,
                    run_mask,
                    None,
                    None,
                    None,
                    [],
                    gamma,
                    **guider_kwargs,
                )
            else:
                if sparse_mask is None:
                    run_mask = (s <= i)
                    not_run_mask = ~run_mask
                    # run_mask = torch.cat([run_mask] * 2)
                    run_mask = torch.where(run_mask)[0]
                    x_scaled_partial = x_scaled[run_mask]
                    # print("run mask:", run_mask)
                    # print("not run mask:", not_run_mask)
                    x_scaled_partial, _ = self.sampler_step_finer(
                        s_in[run_mask] * sigmas[i],
                        s_in[run_mask] * sigmas[i + 1],
                        denoiser,
                        x_scaled_partial,  # ← no manual scaling
                        scale,
                        cond,
                        uc,
                        run_mask,
                        None,
                        None,
                        None,
                        cache_latents.copy(),
                        gamma,
                        **guider_kwargs,
                    )
                    x_scaled[run_mask] = x_scaled_partial
                    x_scaled[not_run_mask] = x_scaled_copy[not_run_mask] + noise[not_run_mask] * sigmas[i+1]
                else:
                    run_mask = (s <= i)
                    not_run_mask = ~run_mask
                    # run_mask = torch.cat([run_mask] * 2)
                    run_mask = torch.where(run_mask)[0]
                    x_scaled_partial = x_scaled[run_mask]
                    # print("run mask:", run_mask)
                    # print("not run mask:", not_run_mask)
                    x_scaled_partial, _ = self.sampler_step_finer(
                        s_in[run_mask] * sigmas[i],
                        s_in[run_mask] * sigmas[i + 1],
                        denoiser,
                        x_scaled_partial,  # ← no manual scaling
                        scale,
                        cond,
                        uc,
                        run_mask,
                        sparse_mask,
                        block_mask,
                        meta,
                        cache_latents.copy(),
                        gamma,
                        **guider_kwargs,
                    )
                    # x_scaled[run_mask] = x_scaled_partial
                    # x_scaled[not_run_mask] = x_scaled_copy[not_run_mask] + noise[not_run_mask] * sigmas[i+1]
                    
                    x_scaled = x_scaled_copy + noise* sigmas[i+1]    
                    tmp = x_scaled[run_mask]
                    tmp = torch.where(mask_hw, x_scaled_partial, tmp)
                    x_scaled[run_mask] = tmp


            # print("x_scaled shape:", x_scaled.shape)


        return x_scaled