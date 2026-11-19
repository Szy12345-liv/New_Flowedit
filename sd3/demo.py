"""
DecFlowEdit demo on Stable Diffusion 3 / 3.5-Medium.

Runs prior extraction followed by prior-guided editing on one image.
Outputs default to outputs/sd3/<schedule> or outputs/sd35/<schedule> in the
package directory, independent of the working directory.

Usage:
    # Stable Diffusion 3 (medium):
    python demo.py --model stabilityai/stable-diffusion-3-medium-diffusers

    # Stable Diffusion 3.5 (medium):
    python demo.py --model stabilityai/stable-diffusion-3.5-medium

Follow paper defaults for SD3 / SD3.5:
    Full prior      = 50 steps, n_max 33 (legacy aggregation indices 17..38)
    Short prior     = 13 steps, n_max 9 (closed aggregation indices 4..9)
    Final editing   = 50 steps, n_max 33 for either prior schedule
    Prior guidance  = (1.0, 1.0)        # w/o CFG
    Edit guidance   = (3.5, 13.5)       # default CFG for SD3 / SD3.5
    tau, softness   = 0.55, 0.08
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))
from common import add_sampling_arguments, resolve_sampling, anonymous_model_reference


DEFAULT_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_IMAGE = str(PACKAGE_DIR / "assets" / "sample.jpg")
DEFAULT_SRC_PROMPT = "a slanted mountain bicycle on the road in front of a building"
DEFAULT_TAR_PROMPT = "a slanted rusty mountain bicycle on the road in front of a building"
DEFAULT_NEGATIVE = ""


def encode_image(pipe, image_path: str, height: int, width: int, device, dtype):
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    img_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    image_tensor = img_processor.preprocess(image, height=height, width=width).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor).latent_dist.mode()
    latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latent


def save_mask_visualizations(mask_dict, output_dir):
    import matplotlib.cm
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    for name, key, cmap in [
        ("E.png", "E", "turbo"),
        ("M_soft.png", "M_soft", "gray"),
        ("M_hard.png", "M_hard", "gray"),
    ]:
        arr = np.clip(mask_dict[key].astype(np.float32), 0.0, 1.0)
        if cmap == "gray":
            img_arr = (arr * 255.0).astype(np.uint8)
            Image.fromarray(img_arr).save(os.path.join(output_dir, name))
        else:
            colored = matplotlib.cm.get_cmap(cmap)(arr)[..., :3]
            img_arr = np.clip(colored * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img_arr).save(os.path.join(output_dir, name))


def load_saved_soft_mask(path):
    """Use the saved uint8 prior, matching both full and short SD experiments."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255.0)


def reset_seed(seed):
    """Start each stage from the requested seed, as in the experiment worker."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF model id; e.g. stabilityai/stable-diffusion-3-medium-diffusers "
                             "or stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--src_prompt", default=DEFAULT_SRC_PROMPT)
    parser.add_argument("--tar_prompt", default=DEFAULT_TAR_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--output_dir", default=None,
                        help="Output directory; defaults to package outputs/<model>/<prior_schedule>.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    add_sampling_arguments(parser, "sd3")
    parser.add_argument("--n_min", type=int, default=0,
                        help="SDEdit-style refinement window: skip paired-state once T_steps - i <= n_min. Inherited from FlowEdit's reference implementation.")
    parser.add_argument("--src_guidance", type=float, default=3.5)
    parser.add_argument("--tar_guidance", type=float, default=13.5)
    parser.add_argument("--tau", type=float, default=0.55)
    parser.add_argument("--softness", type=float, default=0.08)
    parser.add_argument("--agg_low", type=float, default=0.25)
    parser.add_argument("--agg_high", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        sampling = resolve_sampling(args, "sd3")
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 <= args.agg_low <= args.agg_high <= 1.0:
        parser.error("Require 0 <= agg_low <= agg_high <= 1.")

    import torch
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from utils import (
        build_mask_from_velocity_diff_sum,
        decode_latent_to_image,
        run_flowedit_collect_velocity_diff_sum,
        run_flowedit_with_mask,
    )

    model_name = "sd35" if any(
        name in args.model.lower() for name in ("stable-diffusion-3.5", "sd3.5", "sd35")
    ) else "sd3"
    output_dir = Path(args.output_dir) if args.output_dir else (
        PACKAGE_DIR / "outputs" / model_name / args.prior_schedule
    )
    reset_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[1/4] Loading {args.model} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"[2/4] Encoding source image: {args.image}")
    reset_seed(args.seed)
    x_src = encode_image(pipe, args.image, args.height, args.width, device, dtype)

    print(f"[3/4] Prior extraction: {sampling.prior_steps} steps, "
          f"n_max={sampling.prior_n_max}, w/o CFG ...")
    reset_seed(args.seed)
    pass1 = run_flowedit_collect_velocity_diff_sum(
        pipe, scheduler, x_src,
        src_prompt=args.src_prompt, tar_prompt=args.tar_prompt,
        negative_prompt=args.negative_prompt,
        T_steps=sampling.prior_steps,
        src_guidance_scale=1.0, tar_guidance_scale=1.0,
        aggregation_range=(args.agg_low, args.agg_high),
        aggregation_rounding=sampling.aggregation_rounding,
        n_min=args.n_min, n_max=sampling.prior_n_max,
        height=args.height, width=args.width,
    )
    mask_dict = build_mask_from_velocity_diff_sum(
        pass1["velocity_diff_sum"],
        tau=args.tau, softness=args.softness,
    )
    save_mask_visualizations(mask_dict, output_dir)
    run_config = {
        "method": "DecFlowEdit",
        "model": anonymous_model_reference(args.model),
        "prior_schedule": args.prior_schedule,
        "sampling": asdict(sampling),
        "selected_step_indices": pass1["selected_step_indices"],
        "aggregation_range": [args.agg_low, args.agg_high],
        "prior_guidance": [1.0, 1.0],
        "edit_guidance": [args.src_guidance, args.tar_guidance],
        "n_min": args.n_min,
        "tau": args.tau,
        "softness": args.softness,
        "smoothing_kernel": 5,
        "quantiles": [10.0, 90.0],
        "seed": args.seed,
        "seed_reset_per_stage": True,
        "height": args.height,
        "width": args.width,
        "mask_storage": "M_soft.png: uint8 grayscale, reloaded as float32 / 255 for editing",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"    Spatial prior saved to {output_dir}/")

    print(f"[4/4] Prior-guided editing: {sampling.edit_steps} steps, "
          f"n_max={sampling.edit_n_max}, default CFG ...")
    M_soft_tensor = load_saved_soft_mask(output_dir / "M_soft.png")
    reset_seed(args.seed)
    final_latent = run_flowedit_with_mask(
        pipe, scheduler, x_src,
        src_prompt=args.src_prompt, tar_prompt=args.tar_prompt,
        M_soft=M_soft_tensor,
        negative_prompt=args.negative_prompt,
        T_steps=sampling.edit_steps,
        src_guidance_scale=args.src_guidance,
        tar_guidance_scale=args.tar_guidance,
        n_min=args.n_min, n_max=sampling.edit_n_max,
        height=args.height, width=args.width,
    )
    edited = decode_latent_to_image(pipe, final_latent)
    out_path = output_dir / "edited.png"
    edited.save(out_path)
    print(f"Done. Edited image written to {out_path}")


if __name__ == "__main__":
    main()
