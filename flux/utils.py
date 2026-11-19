"""
DecFlowEdit core implementation for FLUX backbone.

This is a minimal, self-contained implementation of the algorithm described in
the paper. The pipeline has two stages:

  Prior-extraction stage (w/o CFG):
     - run_flowedit_collect_velocity_diff_sum: aggregate velocity differences
       Delta V_t = V_tar - V_src across timesteps as VECTORS.
     - build_mask_from_velocity_diff_sum: take channel-wise L2 norm of the
       aggregated vector to obtain E, normalize, and apply sigmoid to
       obtain a soft spatial prior M(x).

  Prior-guided editing stage (default CFG):
     - run_flowedit_with_mask: standard FlowEdit, but at each timestep the
       velocity difference Delta V_t is multiplied spatially by M(x)
       (background regions where M ~ 0 are suppressed).
"""
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


# ===========================================================================
# Low-level helpers (FLUX-specific)
# ===========================================================================

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _scale_noise(scheduler, sample, timestep, noise):
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


@torch.no_grad()
def _calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance,
                text_ids, latent_image_ids, t):
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    latents = latents.to(dtype=transformer_dtype)
    prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=transformer_dtype)
    timestep = t.expand(latents.shape[0])
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]


def _quantile_normalize(energy_map: torch.Tensor, q_low: float = 10.0,
                       q_high: float = 90.0, eps: float = 1e-8) -> torch.Tensor:
    ql = torch.quantile(energy_map, q_low / 100.0)
    qh = torch.quantile(energy_map, q_high / 100.0)
    denom = torch.clamp(qh - ql, min=eps)
    return torch.clamp((energy_map - ql) / denom, 0.0, 1.0)


# ===========================================================================
# Input preparation
# ===========================================================================

def _prepare_inputs(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    T_steps: int,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    height: int,
    width: int,
) -> Dict[str, Any]:
    device = x_src.device
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=height,
        width=width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=height, width=width,
        dtype=x_src.dtype, device=device,
        generator=None, latents=x_src,
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents,
        x_src.shape[2], x_src.shape[3],
    )

    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = _calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device,
        timesteps=None, sigmas=sigmas, mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, device=device,
    )
    pipe._guidance_scale = tar_guidance_scale
    tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt, prompt_2=None, device=device,
    )

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
    else:
        src_guidance = tar_guidance = None

    return {
        "x_src_packed": x_src_packed,
        "zt_edit": x_src_packed.clone(),
        "timesteps": timesteps,
        "T_steps": T_steps,
        "src_prompt_embeds": src_prompt_embeds,
        "src_pooled_prompt_embeds": src_pooled_prompt_embeds,
        "src_text_ids": src_text_ids,
        "tar_prompt_embeds": tar_prompt_embeds,
        "tar_pooled_prompt_embeds": tar_pooled_prompt_embeds,
        "tar_text_ids": tar_text_ids,
        "src_guidance": src_guidance,
        "tar_guidance": tar_guidance,
        "latent_image_ids": latent_image_ids,
        "orig_height": height,
        "orig_width": width,
        "num_channels_latents": num_channels_latents,
    }


# ===========================================================================
# Prior extraction: aggregate velocity differences
# ===========================================================================

@torch.no_grad()
def run_flowedit_collect_velocity_diff_sum(
    pipe, scheduler, x_src,
    src_prompt: str, tar_prompt: str,
    T_steps: int = 28,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 1.0,
    aggregation_range: Tuple[float, float] = (0.25, 0.75),
    aggregation_rounding: str = "legacy",
    n_min: int = 0,
    n_max: int = 24,
    height: int = None, width: int = None,
) -> Dict[str, Any]:
    """Run a FlowEdit trajectory under w/o CFG and aggregate Delta V_t as vectors
    across timesteps in the chosen interval. Returns the running sum (vector
    field) which is then converted to E by build_mask_from_velocity_diff_sum.

    n_max controls SDEdit-style skip of the earliest timesteps (inherited from
    FlowEdit's reference implementation): the loop only becomes active once
    T_steps - i <= n_max (with default T=28, n_max=24, the first 4 timesteps
    are skipped).
    """
    prep = _prepare_inputs(
        pipe, scheduler, x_src, src_prompt, tar_prompt,
        T_steps, src_guidance_scale, tar_guidance_scale, height, width,
    )
    x_src_packed = prep["x_src_packed"]
    zt_edit = prep["zt_edit"]
    timesteps = prep["timesteps"]
    T_steps = prep["T_steps"]

    if aggregation_rounding == "closed":
        # Short schedules follow the inclusive interval used in the experiments.
        agg_low_idx = int(np.ceil(aggregation_range[0] * T_steps))
        agg_high_idx = int(np.floor(aggregation_range[1] * T_steps))
    elif aggregation_rounding == "legacy":
        agg_low_idx = int(np.floor(aggregation_range[0] * T_steps))
        agg_high_idx = int(np.ceil(aggregation_range[1] * T_steps))
    else:
        raise ValueError("aggregation_rounding must be 'legacy' or 'closed'")
    selected_step_indices = []

    velocity_diff_sum: Optional[torch.Tensor] = None

    for i, t in tqdm(enumerate(timesteps), desc="Prior extraction: aggregate velocity differences"):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        # Preserve the experiment implementation: the final iteration has zero dt.
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if T_steps - i > n_min:
            fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
            zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise
            zt_tar = zt_edit + zt_src - x_src_packed

            Vt_src = _calc_v_flux(
                pipe, latents=zt_src,
                prompt_embeds=prep["src_prompt_embeds"],
                pooled_prompt_embeds=prep["src_pooled_prompt_embeds"],
                guidance=prep["src_guidance"],
                text_ids=prep["src_text_ids"],
                latent_image_ids=prep["latent_image_ids"], t=t,
            )
            Vt_tar = _calc_v_flux(
                pipe, latents=zt_tar,
                prompt_embeds=prep["tar_prompt_embeds"],
                pooled_prompt_embeds=prep["tar_pooled_prompt_embeds"],
                guidance=prep["tar_guidance"],
                text_ids=prep["tar_text_ids"],
                latent_image_ids=prep["latent_image_ids"], t=t,
            )
            V_delta = Vt_tar - Vt_src

            if agg_low_idx <= i <= agg_high_idx:
                V_delta_unpacked = pipe._unpack_latents(
                    V_delta, prep["orig_height"], prep["orig_width"], pipe.vae_scale_factor,
                ).float()
                contribution = V_delta_unpacked[0].detach().cpu()
                if velocity_diff_sum is None:
                    velocity_diff_sum = torch.zeros_like(contribution)
                velocity_diff_sum.add_(contribution)
                selected_step_indices.append(int(i))

            zt_edit = zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta
            zt_edit = zt_edit.to(V_delta.dtype)

    return {"velocity_diff_sum": velocity_diff_sum,
            "selected_step_indices": selected_step_indices}


def build_mask_from_velocity_diff_sum(
    velocity_diff_sum: torch.Tensor,
    smoothing_kernel: int = 5,
    tau: float = 0.55,
    softness: float = 0.08,
    q_low: float = 10.0,
    q_high: float = 90.0,
) -> Dict[str, np.ndarray]:
    """Convert aggregated vector field into a soft spatial prior M(x).

      E = ||sum_t Delta V_t||_2     (channel-wise norm of the AGGREGATED sum)
      E_norm = quantile_normalize(E, q_low, q_high)
      M_soft = sigmoid((E_norm - tau) / softness)
    """
    if velocity_diff_sum is None or velocity_diff_sum.dim() != 3:
        raise ValueError("velocity_diff_sum must be a [C, H, W] tensor")

    sum_tensor = velocity_diff_sum.float()
    E_raw = torch.linalg.vector_norm(sum_tensor, dim=0)  # [H, W]

    if smoothing_kernel and smoothing_kernel > 1:
        pad = smoothing_kernel // 2
        E_raw = F.avg_pool2d(
            E_raw.unsqueeze(0).unsqueeze(0),
            kernel_size=smoothing_kernel, stride=1, padding=pad,
        ).squeeze(0).squeeze(0)

    E_norm = _quantile_normalize(E_raw, q_low=q_low, q_high=q_high)
    M_soft = torch.sigmoid((E_norm - tau) / max(softness, 1e-6))
    M_hard = (M_soft > 0.5).float()

    return {
        "E": E_norm.cpu().numpy().astype(np.float32),
        "M_soft": M_soft.cpu().numpy().astype(np.float32),
        "M_hard": M_hard.cpu().numpy().astype(np.float32),
    }


# ===========================================================================
# Prior-guided editing with default CFG
# ===========================================================================

@torch.no_grad()
def run_flowedit_with_mask(
    pipe, scheduler, x_src,
    src_prompt: str, tar_prompt: str,
    M_soft: torch.Tensor,
    T_steps: int = 28,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    height: int = None, width: int = None,
) -> torch.Tensor:
    """Run FlowEdit, applying spatial reweighting Delta V_t <- M(x) * Delta V_t
    at each timestep. Returns final unpacked latent ready for VAE decoding.

    n_max controls SDEdit-style skip of the earliest timesteps (inherited
    from FlowEdit's reference implementation): the loop only becomes active
    once T_steps - i <= n_max.
    """
    prep = _prepare_inputs(
        pipe, scheduler, x_src, src_prompt, tar_prompt,
        T_steps, src_guidance_scale, tar_guidance_scale, height, width,
    )
    x_src_packed = prep["x_src_packed"]
    zt_edit = prep["zt_edit"]
    timesteps = prep["timesteps"]
    T_steps = prep["T_steps"]
    orig_h, orig_w = prep["orig_height"], prep["orig_width"]
    num_channels_latents = prep["num_channels_latents"]

    model_dtype = next(pipe.transformer.parameters()).dtype
    src_prompt_embeds = prep["src_prompt_embeds"].to(model_dtype)
    src_pooled = prep["src_pooled_prompt_embeds"].to(model_dtype)
    src_text_ids = prep["src_text_ids"].to(model_dtype)
    tar_prompt_embeds = prep["tar_prompt_embeds"].to(model_dtype)
    tar_pooled = prep["tar_pooled_prompt_embeds"].to(model_dtype)
    tar_text_ids = prep["tar_text_ids"].to(model_dtype)
    latent_ids = prep["latent_image_ids"].to(model_dtype)

    M_soft_cpu = M_soft.float().cpu()

    for i, t in tqdm(enumerate(timesteps), desc="Prior-guided editing"):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        # Preserve the experiment implementation: the final iteration has zero dt.
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if T_steps - i > n_min:
            fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
            zt_src = ((1 - t_i) * x_src_packed + t_i * fwd_noise).to(model_dtype)
            zt_tar = (zt_edit + zt_src - x_src_packed).to(model_dtype)

            Vt_src = _calc_v_flux(
                pipe, zt_src, src_prompt_embeds, src_pooled,
                prep["src_guidance"], src_text_ids, latent_ids, t,
            )
            Vt_tar = _calc_v_flux(
                pipe, zt_tar, tar_prompt_embeds, tar_pooled,
                prep["tar_guidance"], tar_text_ids, latent_ids, t,
            )
            V_delta = Vt_tar - Vt_src

            # apply spatial reweighting in unpacked (image-shaped) form
            V_delta_unpacked = pipe._unpack_latents(V_delta, orig_h, orig_w, pipe.vae_scale_factor)
            B, C, H_lat, W_lat = V_delta_unpacked.shape
            mask = M_soft_cpu.to(device=V_delta_unpacked.device, dtype=V_delta_unpacked.dtype)
            if mask.shape != (H_lat, W_lat):
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H_lat, W_lat), mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0).to(V_delta_unpacked.dtype)
            V_delta_unpacked = V_delta_unpacked * mask[None, None, :, :]
            V_delta = pipe._pack_latents(V_delta_unpacked, B, num_channels_latents, H_lat, W_lat)
            V_delta = V_delta.to(x_src_packed.dtype)

            zt_edit = zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta
            zt_edit = zt_edit.to(x_src_packed.dtype)
        else:
            # SDEdit-style refinement: skip paired-state and roll out target only
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = _scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = (zt_edit + xt_src - x_src_packed).to(model_dtype)
            Vt_tar = _calc_v_flux(
                pipe, xt_tar, tar_prompt_embeds, tar_pooled,
                prep["tar_guidance"], tar_text_ids, latent_ids, t,
            )
            xt_tar = xt_tar.to(torch.float32) + (t_im1 - t_i) * Vt_tar
            xt_tar = xt_tar.to(Vt_tar.dtype)
            zt_edit = xt_tar

    return pipe._unpack_latents(zt_edit, orig_h, orig_w, pipe.vae_scale_factor)


# ===========================================================================
# Decode
# ===========================================================================

@torch.no_grad()
def decode_latent_to_image(pipe, latent: torch.Tensor) -> Image.Image:
    """Decode an unpacked FLUX latent [B, C, H_lat, W_lat] into a PIL image."""
    latent = latent.to(pipe.device)
    denorm = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    autocast_ctx = torch.autocast("cuda") if pipe.device.type == "cuda" else nullcontext()
    with autocast_ctx, torch.inference_mode():
        decoded = pipe.vae.decode(denorm, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded)[0]
