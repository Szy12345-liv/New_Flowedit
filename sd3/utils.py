"""
DecFlowEdit core implementation for Stable Diffusion 3 / 3.5-Medium backbones.

This is a minimal, self-contained implementation of the algorithm. The pipeline
has two stages:

  Prior extraction (w/o CFG):
     - run_flowedit_collect_velocity_diff_sum: aggregate velocity differences
       Delta V_t = V_tar - V_src across timesteps as VECTORS.
     - build_mask_from_velocity_diff_sum: take channel-wise L2 norm of the
       aggregated vector to obtain E_coh, normalize, and apply sigmoid to
       obtain a soft spatial prior M(x).

  Prior-guided editing (default CFG):
     - run_flowedit_with_mask: standard FlowEdit, but at each timestep the
       differential update Delta V_t is multiplied spatially by M(x)
       (updates at locations where M ~ 0 are suppressed).

SD3-specific notes:
  * Latents are kept in standard [B, C, H_lat, W_lat] form (no FLUX packing).
  * At guidance (1, 1), only the source and target conditional predictions
    are evaluated, jointly as a batch of 2 ([src_pos, tar_pos]).
  * Other guidance settings use classical CFG with a batch of 4
    ([src_neg, src_pos, tar_neg, tar_pos]).
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
# Low-level helpers (SD3-specific)
# ===========================================================================

def _scale_noise(scheduler, sample, timestep, noise):
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


@torch.no_grad()
def _calc_v_sd3(pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t):
    """Return source and target velocities from a joint MM-DiT forward.

    Guidance (1, 1) uses two conditional branches; other settings mix four
    conditional/unconditional branches with the requested guidance scales.
    """
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    noise_pred = pipe.transformer(
        hidden_states=src_tar_latent_model_input,
        timestep=timestep,
        encoder_hidden_states=src_tar_prompt_embeds,
        pooled_projections=src_tar_pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    if pipe.do_classifier_free_guidance:
        src_neg, src_pos, tar_neg, tar_pos = noise_pred.chunk(4)
        v_src = src_neg + src_guidance_scale * (src_pos - src_neg)
        v_tar = tar_neg + tar_guidance_scale * (tar_pos - tar_neg)
    else:
        v_src, v_tar = noise_pred.chunk(2)
    return v_src, v_tar


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
    negative_prompt: str,
    T_steps: int,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    height: int,
    width: int,
) -> Dict[str, Any]:
    device = x_src.device
    orig_height = height if height is not None else x_src.shape[2] * pipe.vae_scale_factor
    orig_width = width if width is not None else x_src.shape[3] * pipe.vae_scale_factor

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    # At (1, 1), CFG reduces exactly to the conditional prediction.
    # Keep explicit CFG for every other setting, including scales below 1.
    use_cfg = not (src_guidance_scale == 1.0 and tar_guidance_scale == 1.0)
    pipe._guidance_scale = (
        max(src_guidance_scale, tar_guidance_scale, 1.01) if use_cfg else 1.0
    )
    (src_pe, src_neg_pe, src_pool, src_neg_pool) = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=use_cfg,
        device=device,
    )
    (tar_pe, tar_neg_pe, tar_pool, tar_neg_pool) = pipe.encode_prompt(
        prompt=tar_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=use_cfg,
        device=device,
    )

    if use_cfg:
        # Order: [src_neg, src_pos, tar_neg, tar_pos].
        src_tar_prompt_embeds = torch.cat([src_neg_pe, src_pe, tar_neg_pe, tar_pe], dim=0)
        src_tar_pooled = torch.cat([src_neg_pool, src_pool, tar_neg_pool, tar_pool], dim=0)
    else:
        # Negative embeddings are not encoded when CFG is disabled.
        src_tar_prompt_embeds = torch.cat([src_pe, tar_pe], dim=0)
        src_tar_pooled = torch.cat([src_pool, tar_pool], dim=0)

    return {
        "x_src": x_src,
        "zt_edit": x_src.clone(),
        "timesteps": timesteps,
        "T_steps": T_steps,
        "src_tar_prompt_embeds": src_tar_prompt_embeds,
        "src_tar_pooled_prompt_embeds": src_tar_pooled,
        "src_guidance_scale": src_guidance_scale,
        "tar_guidance_scale": tar_guidance_scale,
        "orig_height": orig_height,
        "orig_width": orig_width,
    }


# ===========================================================================
# Prior extraction via vector aggregation
# ===========================================================================

@torch.no_grad()
def run_flowedit_collect_velocity_diff_sum(
    pipe, scheduler, x_src,
    src_prompt: str, tar_prompt: str,
    negative_prompt: str = "",
    T_steps: int = 50,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 1.0,
    aggregation_range: Tuple[float, float] = (0.25, 0.75),
    n_min: int = 0,
    n_max: int = 33,
    height: int = None, width: int = None,
    aggregation_rounding: str = "legacy",
) -> Dict[str, Any]:
    """Run SD3 FlowEdit w/o CFG and aggregate Delta V_t as vectors
    across timesteps in the chosen interval. Returns the running vector sum.

    n_max controls SDEdit-style skip of the earliest timesteps (inherited
    from FlowEdit's reference implementation for SD3/SD3.5): the loop only
    becomes active once T_steps - i <= n_max (with default T=50, n_max=33,
    the first 17 timesteps are skipped).

    Legacy full extraction includes floor(.25*T)..ceil(.75*T); closed short
    extraction includes ceil(.25*T)..floor(.75*T). Both intersect active steps.
    """
    prep = _prepare_inputs(
        pipe, scheduler, x_src, src_prompt, tar_prompt, negative_prompt,
        T_steps, src_guidance_scale, tar_guidance_scale, height, width,
    )
    x_src_raw = prep["x_src"]
    zt_edit = prep["zt_edit"]
    timesteps = prep["timesteps"]
    T_steps = prep["T_steps"]
    src_tar_pe = prep["src_tar_prompt_embeds"]
    src_tar_pool = prep["src_tar_pooled_prompt_embeds"]
    src_gs = prep["src_guidance_scale"]
    tar_gs = prep["tar_guidance_scale"]

    if aggregation_rounding == "legacy":
        agg_low_idx = int(np.floor(aggregation_range[0] * T_steps))
        agg_high_idx = int(np.ceil(aggregation_range[1] * T_steps))
    elif aggregation_rounding == "closed":
        agg_low_idx = int(np.ceil(aggregation_range[0] * T_steps))
        agg_high_idx = int(np.floor(aggregation_range[1] * T_steps))
    else:
        raise ValueError("aggregation_rounding must be 'legacy' or 'closed'")

    velocity_diff_sum: Optional[torch.Tensor] = None
    selected_step_indices = []

    for i, t in tqdm(enumerate(timesteps), desc="Prior extraction: aggregate Delta V_t (SD3)"):
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        t_im1 = timesteps[i + 1] / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:
            fwd_noise = torch.randn_like(x_src_raw).to(x_src_raw.device)
            zt_src = (1 - t_i) * x_src_raw + t_i * fwd_noise
            zt_tar = zt_edit + zt_src - x_src_raw

            src_tar_latent = (
                torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                if pipe.do_classifier_free_guidance else torch.cat([zt_src, zt_tar])
            )
            Vt_src, Vt_tar = _calc_v_sd3(
                pipe, src_tar_latent, src_tar_pe, src_tar_pool, src_gs, tar_gs, t,
            )
            V_delta = Vt_tar - Vt_src

            if agg_low_idx <= i <= agg_high_idx:
                contribution = V_delta[0].float().detach().cpu()
                if velocity_diff_sum is None:
                    velocity_diff_sum = torch.zeros_like(contribution)
                velocity_diff_sum.add_(contribution)
                selected_step_indices.append(i)

            zt_edit = zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta
            zt_edit = zt_edit.to(V_delta.dtype)

    return {
        "velocity_diff_sum": velocity_diff_sum,
        "selected_step_indices": selected_step_indices,
        "aggregation_rounding": aggregation_rounding,
    }


def build_mask_from_velocity_diff_sum(
    velocity_diff_sum: torch.Tensor,
    smoothing_kernel: int = 5,
    tau: float = 0.55,
    softness: float = 0.08,
    q_low: float = 10.0,
    q_high: float = 90.0,
) -> Dict[str, np.ndarray]:
    """Convert aggregated vector field into a soft spatial prior M(x)."""
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
    negative_prompt: str = "",
    T_steps: int = 50,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 33,
    height: int = None, width: int = None,
) -> torch.Tensor:
    """Run SD3 FlowEdit, applying spatial reweighting Delta V_t <- M(x) * Delta V_t
    at each timestep. Returns the final latent [B, C, H_lat, W_lat] (not packed).

    n_max controls SDEdit-style skip of the earliest timesteps (inherited
    from FlowEdit's reference implementation for SD3/SD3.5).
    """
    prep = _prepare_inputs(
        pipe, scheduler, x_src, src_prompt, tar_prompt, negative_prompt,
        T_steps, src_guidance_scale, tar_guidance_scale, height, width,
    )
    x_src_raw = prep["x_src"]
    zt_edit = prep["zt_edit"]
    timesteps = prep["timesteps"]
    T_steps = prep["T_steps"]
    src_tar_pe = prep["src_tar_prompt_embeds"]
    src_tar_pool = prep["src_tar_pooled_prompt_embeds"]
    src_gs = prep["src_guidance_scale"]
    tar_gs = prep["tar_guidance_scale"]

    M_soft_cpu = M_soft.float().cpu()

    for i, t in tqdm(enumerate(timesteps), desc="Prior-guided editing (SD3)"):
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        t_im1 = timesteps[i + 1] / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:
            fwd_noise = torch.randn_like(x_src_raw).to(x_src_raw.device)
            zt_src = (1 - t_i) * x_src_raw + t_i * fwd_noise
            zt_tar = zt_edit + zt_src - x_src_raw

            src_tar_latent = (
                torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                if pipe.do_classifier_free_guidance else torch.cat([zt_src, zt_tar])
            )
            Vt_src, Vt_tar = _calc_v_sd3(
                pipe, src_tar_latent, src_tar_pe, src_tar_pool, src_gs, tar_gs, t,
            )
            V_delta = Vt_tar - Vt_src

            B, C, H_lat, W_lat = V_delta.shape
            mask = M_soft_cpu.to(device=V_delta.device, dtype=V_delta.dtype)
            if mask.shape != (H_lat, W_lat):
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H_lat, W_lat), mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0).to(V_delta.dtype)
            V_delta = V_delta * mask[None, None, :, :]

            zt_edit = zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta
            zt_edit = zt_edit.to(V_delta.dtype)
        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_raw).to(x_src_raw.device)
                xt_src = _scale_noise(scheduler, x_src_raw, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_raw
            src_tar_latent = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance else torch.cat([xt_tar, xt_tar])
            )
            _, Vt_tar = _calc_v_sd3(
                pipe, src_tar_latent, src_tar_pe, src_tar_pool, src_gs, tar_gs, t,
            )
            xt_tar = xt_tar.to(torch.float32) + (t_im1 - t_i) * Vt_tar
            xt_tar = xt_tar.to(Vt_tar.dtype)
            zt_edit = xt_tar

    return zt_edit


# ===========================================================================
# Decode
# ===========================================================================

@torch.no_grad()
def decode_latent_to_image(pipe, latent: torch.Tensor) -> Image.Image:
    """Decode an SD3 latent [B, C, H_lat, W_lat] to a PIL image."""
    latent = latent.to(pipe.device)
    denorm = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    autocast_ctx = torch.autocast("cuda") if pipe.device.type == "cuda" else nullcontext()
    with autocast_ctx, torch.inference_mode():
        decoded = pipe.vae.decode(denorm, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded)[0]
