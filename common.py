"""Sampling presets shared by the DecFlowEdit demonstration entry points."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingConfig:
    prior_steps: int
    prior_n_max: int
    edit_steps: int
    edit_n_max: int
    aggregation_rounding: str


def add_sampling_arguments(parser, backbone):
    """Keep prior extraction and final editing independently configurable."""
    if backbone not in ("flux", "sd3"):
        raise ValueError("backbone must be 'flux' or 'sd3'")
    steps, n_max = (28, 24) if backbone == "flux" else (50, 33)
    parser.add_argument("--prior_schedule", choices=("full", "short"), default="full",
                        help="Full or approximately quarter-length prior extraction.")
    parser.add_argument("--prior_steps", type=int, default=None,
                        help="Override the prior-extraction schedule length only.")
    parser.add_argument("--prior_n_max", type=int, default=None,
                        help="Override the active prior-extraction step count only.")
    parser.add_argument("--edit_steps", "--T_steps", dest="edit_steps", type=int,
                        default=steps, help="Final editing schedule length.")
    parser.add_argument("--edit_n_max", "--n_max", dest="edit_n_max", type=int,
                        default=n_max, help="Active final editing step count.")


def resolve_sampling(args, backbone):
    """Resolve full/short presets, retaining the executed experiment windows.

    With the default editing schedules, short resolves to 7/6 for FLUX
    and 13/9 for SD3 or SD3.5 (schedule length / n_max).
    """
    if backbone not in ("flux", "sd3"):
        raise ValueError("backbone must be 'flux' or 'sd3'")
    edit_steps, edit_n_max = args.edit_steps, args.edit_n_max
    if edit_steps <= 0 or not 0 < edit_n_max <= edit_steps:
        raise ValueError("Require 0 < edit_n_max <= edit_steps.")
    prior_steps = args.prior_steps
    if prior_steps is None:
        prior_steps = edit_steps if args.prior_schedule == "full" else math.ceil(edit_steps / 4)
    if prior_steps <= 0:
        raise ValueError("prior_steps must be positive.")
    prior_n_max = args.prior_n_max
    if prior_n_max is None:
        prior_n_max = math.ceil(edit_n_max * prior_steps / edit_steps)
    if not 0 < prior_n_max <= prior_steps:
        raise ValueError("Require 0 < prior_n_max <= prior_steps.")
    n_min = getattr(args, "n_min", 0)
    if not 0 <= n_min < min(prior_n_max, edit_n_max):
        raise ValueError("n_min must be nonnegative and smaller than both n_max values.")
    rounding = "legacy" if args.prior_schedule == "full" else "closed"
    return SamplingConfig(prior_steps, prior_n_max, edit_steps, edit_n_max, rounding)


def anonymous_model_reference(model):
    """Only public official IDs are included in the shareable run record."""
    official_ids = {
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
    }
    value = str(model)
    return value if value in official_ids else "custom_checkpoint"
