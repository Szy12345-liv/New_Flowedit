"""DecFlowEdit single-image demo on FLUX.1-dev.

Prior extraction uses w/o CFG (1, 1), with a full or short sampling schedule.
Prior-guided editing uses default CFG (1.5, 5.5) and 28 scheduled steps by default.
The soft prior is saved as an 8-bit PNG and reloaded for editing, matching the
batch experiment path. The existing FLUX final-iteration rule is retained.
"""
import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from common import add_sampling_arguments, resolve_sampling, anonymous_model_reference

DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_IMAGE = SCRIPT_DIR.parent / "assets" / "sample.jpg"
DEFAULT_SRC_PROMPT = "a slanted mountain bicycle on the road in front of a building"
DEFAULT_TAR_PROMPT = "a slanted rusty mountain bicycle on the road in front of a building"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "flux"


def encode_image(pipe, image_path, height, width, device, dtype):
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    img_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor * 2)
    image_tensor = img_processor.preprocess(image, height=height, width=width).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor).latent_dist.mode()
    return (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


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
            Image.fromarray((arr * 255.0).astype(np.uint8)).save(Path(output_dir) / name)
        else:
            colored = matplotlib.cm.get_cmap(cmap)(arr)[..., :3]
            Image.fromarray(np.clip(colored * 255.0, 0, 255).astype(np.uint8)).save(Path(output_dir) / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--src_prompt", default=DEFAULT_SRC_PROMPT)
    parser.add_argument("--tar_prompt", default=DEFAULT_TAR_PROMPT)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output folder; default: package outputs/flux/<prior_schedule>.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    add_sampling_arguments(parser, "flux")
    parser.add_argument("--n_min", type=int, default=0,
                        help="FlowEdit target-only refinement window; the experiments use 0.")
    parser.add_argument("--src_guidance", type=float, default=1.5)
    parser.add_argument("--tar_guidance", type=float, default=5.5)
    parser.add_argument("--tau", type=float, default=0.55)
    parser.add_argument("--softness", type=float, default=0.08)
    parser.add_argument("--agg_low", type=float, default=0.25)
    parser.add_argument("--agg_high", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / args.prior_schedule
    try:
        sampling = resolve_sampling(args, "flux")
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.agg_low < args.agg_high <= 1:
        parser.error("aggregation bounds must satisfy 0 <= agg_low < agg_high <= 1")
    if not 0 <= args.n_min < min(sampling.prior_n_max, sampling.edit_n_max):
        parser.error("n_min must be nonnegative and smaller than both stages' n_max")
    if args.softness <= 0:
        parser.error("softness must be positive")
    if not args.image.is_file():
        parser.error("input image does not exist: " + str(args.image))

    # Keep --help available without importing model packages or loading weights.
    import numpy as np
    import random
    import torch
    from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
    from PIL import Image
    from utils import (
        build_mask_from_velocity_diff_sum,
        decode_latent_to_image,
        run_flowedit_collect_velocity_diff_sum,
        run_flowedit_with_mask,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    def reset_seed():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    reset_seed()
    print(f"[1/4] Loading {args.model} ...")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"[2/4] Encoding source image: {args.image}")
    x_src = encode_image(pipe, args.image, args.height, args.width, device, dtype)
    print(f"[3/4] Prior extraction (w/o CFG): {sampling.prior_steps} scheduled steps ...")
    prior_result = run_flowedit_collect_velocity_diff_sum(
        pipe, scheduler, x_src,
        src_prompt=args.src_prompt, tar_prompt=args.tar_prompt,
        T_steps=sampling.prior_steps,
        src_guidance_scale=1.0, tar_guidance_scale=1.0,
        aggregation_range=(args.agg_low, args.agg_high),
        aggregation_rounding=sampling.aggregation_rounding,
        n_min=args.n_min, n_max=sampling.prior_n_max,
        height=args.height, width=args.width,
    )
    mask_dict = build_mask_from_velocity_diff_sum(
        prior_result["velocity_diff_sum"], tau=args.tau, softness=args.softness,
    )
    save_mask_visualizations(mask_dict, args.output_dir)
    # Match the saved-mask path used for the FLUX batch editing experiments.
    with Image.open(args.output_dir / "M_soft.png") as image:
        M_soft_tensor = torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255.0)

    print(f"[4/4] Prior-guided editing (default CFG): {sampling.edit_steps} scheduled steps ...")
    final_latent = run_flowedit_with_mask(
        pipe, scheduler, x_src,
        src_prompt=args.src_prompt, tar_prompt=args.tar_prompt,
        M_soft=M_soft_tensor,
        T_steps=sampling.edit_steps,
        src_guidance_scale=args.src_guidance,
        tar_guidance_scale=args.tar_guidance,
        n_min=args.n_min, n_max=sampling.edit_n_max,
        height=args.height, width=args.width,
    )
    edited = decode_latent_to_image(pipe, final_latent)
    out_path = args.output_dir / "edited.png"
    edited.save(out_path)
    record = {
        "method": "DecFlowEdit", "backbone": "FLUX",
        "model": anonymous_model_reference(args.model),
        "prior_schedule": args.prior_schedule,
        "prior_steps": sampling.prior_steps, "prior_n_max": sampling.prior_n_max,
        "edit_steps": sampling.edit_steps, "edit_n_max": sampling.edit_n_max,
        "aggregation_rounding": sampling.aggregation_rounding,
        "selected_step_indices": prior_result["selected_step_indices"],
        "prior_guidance": [1.0, 1.0], "edit_guidance": [args.src_guidance, args.tar_guidance],
        "tau": args.tau, "softness": args.softness, "seed": args.seed,
        "seed_policy": "reset once before the two-stage demo; stages share the RNG stream",
        "dtype": str(dtype), "mask_storage": "uint8 PNG reloaded as float32 / 255",
        "final_iteration": "retain the existing FLUX zero-delta-t rule",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Done. Edited image written to {out_path}")


if __name__ == "__main__":
    main()
