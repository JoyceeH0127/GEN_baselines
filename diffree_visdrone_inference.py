#!/usr/bin/env python3
"""Batch Diffree inference for a VisDrone insertion schedule.

Each positive class count is expanded into individual objects. Diffree inserts
them sequentially, using the composited result from one insertion as the input
to the next. The prompt template matches the FLUX/Qwen comparison scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageFilter, ImageOps
from torch import autocast


DEFAULT_DATA_ROOT = Path("/home/qinma/yelo/datasets/Visdrone")
DEFAULT_DIFFREE_ROOT = Path(
    os.environ.get("DIFFREE_ROOT", "/home/qinma/yelo/models/Diffree")
).expanduser()
DEFAULT_OUTPUT_ROOT = Path("/home/qinma/yelo/outputs/Diffree_Visdrone")
DEFAULT_CONFIG_PATH = DEFAULT_DIFFREE_ROOT / "config/generate.yaml"
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_DIFFREE_ROOT / "checkpoints/diffree-step=000010999.ckpt"
)

# Make Diffree's bundled ``stable_diffusion`` package importable even when this
# script is launched from another working directory.
if not (DEFAULT_DIFFREE_ROOT / "stable_diffusion").is_dir():
    raise FileNotFoundError(
        "Diffree repository not found at "
        f"{DEFAULT_DIFFREE_ROOT}. Set DIFFREE_ROOT to the repository path."
    )
sys.path.insert(0, str(DEFAULT_DIFFREE_ROOT))
from stable_diffusion.ldm.util import instantiate_from_config  # noqa: E402


CLASS_NAMES = (
    ("pedestrian", "pedestrian"),
    ("people", "person"),
    ("bicycle", "bicycle"),
    ("car", "car"),
    ("van", "van"),
    ("truck", "truck"),
    ("tricycle", "tricycle"),
    ("awning-tricycle", "awning-tricycle"),
    ("bus", "bus"),
    ("motor", "motor"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--text-cfg-scale", type=float, default=7.5)
    parser.add_argument("--image-cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--mask-dilate-iterations", type=int, default=3)
    parser.add_argument("--mask-blur-radius", type=float, default=3.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-generated-size", action="store_true")
    return parser.parse_args()


def flatten_tasks(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for tasks in schedule["sequences"].values() for task in tasks]


def source_for(root: Path, task: dict[str, Any]) -> Path:
    return root / f"{task['frame_stem']}.jpg"


def destination_for(root: Path, task: dict[str, Any]) -> Path:
    return root / f"{task['frame_stem']}.jpg"


def objects_for(task: dict[str, Any]) -> list[str]:
    counts = task["class_names"]
    objects: list[str] = []
    for json_name, prompt_name in CLASS_NAMES:
        count = int(counts.get(json_name, 0))
        if count < 0:
            raise ValueError(f"Negative object count for {json_name}: {count}")
        objects.extend([prompt_name] * count)
    if not objects:
        raise ValueError(f"No objects requested: {counts}")
    return objects


def prompt_for_object(object_name: str) -> str:
    return (
        "Preserve all existng objects and keep the rest of image unchanged. "
        f"Insert additional 1 {object_name} in aerial view. "
        "Match the aerial perspective, lighting, and style of the scene. "
    )


def write_manifest(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_from_config(
    config: Any,
    checkpoint: Path,
    vae_checkpoint: Path | None = None,
) -> nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Loading model from {checkpoint}")
    payload = load_checkpoint(checkpoint)
    if "global_step" in payload:
        print(f"Global step: {payload['global_step']}")
    state_dict = payload["state_dict"]

    if vae_checkpoint is not None:
        if not vae_checkpoint.is_file():
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")
        vae_state = load_checkpoint(vae_checkpoint)["state_dict"]
        state_dict = {
            key: vae_state[key[len("first_stage_model.") :]]
            if key.startswith("first_stage_model.")
            else value
            for key, value in state_dict.items()
        }

    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model


def append_dims(tensor: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims_to_append = target_dims - tensor.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"Input has {tensor.ndim} dimensions, target has {target_dims}"
        )
    return tensor[(...,) + (None,) * dims_to_append]


class CompVisDenoiser(K.external.CompVisDenoiser):
    def get_eps(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner_model.apply_model(*args, **kwargs)

    def forward(
        self,
        input_image: torch.Tensor,
        input_mask: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_mask
        c_out, c_in = [
            append_dims(value, input_image.ndim) for value in self.get_scalings(sigma)
        ]
        eps_image, eps_mask = self.get_eps(
            input_image * c_in,
            self.sigma_to_t(sigma).to(input_image.device),
            **kwargs,
        )
        return input_image + eps_image * c_out, eps_mask


class CFGDenoiser(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.inner_model = model

    def forward(
        self,
        image_latent: torch.Tensor,
        mask_latent: torch.Tensor,
        sigma: torch.Tensor,
        cond: dict[str, list[torch.Tensor]],
        uncond: dict[str, list[torch.Tensor]],
        text_cfg_scale: float,
        image_cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg_image = einops.repeat(image_latent, "1 ... -> n ...", n=3)
        cfg_mask = einops.repeat(mask_latent, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [
                torch.cat(
                    [
                        cond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                    ]
                )
            ],
            "c_concat": [
                torch.cat(
                    [
                        cond["c_concat"][0],
                        cond["c_concat"][0],
                        uncond["c_concat"][0],
                    ]
                )
            ],
        }
        output_image, output_mask = self.inner_model(
            cfg_image, cfg_mask, cfg_sigma, cond=cfg_cond
        )
        cond_image, image_cond_image, uncond_image = output_image.chunk(3)
        cond_mask, _, _ = output_mask.chunk(3)
        guided_image = (
            uncond_image
            + text_cfg_scale * (cond_image - image_cond_image)
            + image_cfg_scale * (image_cond_image - uncond_image)
        )
        return guided_image, cond_mask


def to_derivative(
    latent: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor
) -> torch.Tensor:
    return (latent - denoised) / append_dims(sigma, latent.ndim)


def ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eta == 0:
        return sigma_to, torch.zeros_like(sigma_to)
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (
            sigma_to**2
            * (sigma_from**2 - sigma_to**2)
            / sigma_from**2
        )
        ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def sample_euler_ancestral(
    model: nn.Module,
    image_latent: torch.Tensor,
    mask_latent: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_batch = image_latent.new_ones([image_latent.shape[0]])
    for step in range(len(sigmas) - 1):
        denoised_image, denoised_mask = model(
            image_latent,
            mask_latent,
            sigmas[step] * sigma_batch,
            **extra_args,
        )
        sigma_down, sigma_up = ancestral_step(sigmas[step], sigmas[step + 1])
        derivative = to_derivative(image_latent, sigmas[step], denoised_image)
        image_latent = image_latent + derivative * (sigma_down - sigmas[step])
        if sigmas[step + 1] > 0:
            image_latent = image_latent + torch.randn_like(image_latent) * sigma_up
        mask_latent = denoised_mask
    return image_latent, mask_latent


class DiffreeRunner:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        vae_checkpoint: Path | None,
        device: torch.device,
    ) -> None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(
            config, checkpoint_path, vae_checkpoint
        ).eval().to(device)
        self.model_wrap = CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""]).to(device)
        self.device = device

    def insert_once(
        self,
        input_image: Image.Image,
        prompt: str,
        seed: int,
        steps: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
        mask_dilate_iterations: int,
        mask_blur_radius: float,
    ) -> tuple[Image.Image, float]:
        source = input_image.convert("RGB")
        source_array = np.asarray(source).astype(np.float32) / 255.0

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            source_tensor = 2 * torch.tensor(np.asarray(source)).float() / 255 - 1
            source_tensor = rearrange(source_tensor, "h w c -> 1 c h w").to(
                self.device
            )
            cond = {
                "c_crossattn": [
                    self.model.get_learned_conditioning([prompt]).to(self.device)
                ],
                "c_concat": [
                    self.model.encode_first_stage(source_tensor).mode().to(self.device)
                ],
            }
            uncond = {
                "c_crossattn": [self.null_token],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])],
            }
            sigmas = self.model_wrap.get_sigmas(steps).to(self.device)
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": text_cfg_scale,
                "image_cfg_scale": image_cfg_scale,
            }
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            image_latent = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            mask_latent = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            image_latent, mask_latent = sample_euler_ancestral(
                self.model_wrap_cfg,
                image_latent,
                mask_latent,
                sigmas,
                extra_args,
            )
            generated = self.model.decode_first_stage(image_latent)
            mask_latent = nn.functional.interpolate(
                mask_latent,
                size=(source.height, source.width),
                mode="bilinear",
                align_corners=False,
            )
            binary_mask = torch.where(mask_latent > 0, 1, -1)
            mask_mean = float(binary_mask.float().mean().item())
            generated = torch.clamp((generated + 1) / 2, 0, 1)
            binary_mask = torch.clamp((binary_mask + 1) / 2, 0, 1)
            generated_array = rearrange(
                generated, "1 c h w -> h w c"
            ).float().cpu().numpy()
            mask_array = rearrange(
                binary_mask, "1 c h w -> h w c"
            ).float().cpu().numpy()

        mask_array = np.repeat(mask_array[:, :, :1], 3, axis=2)
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = (mask_array * 255).astype(np.uint8)
        mask_uint8 = cv2.dilate(
            mask_uint8, kernel, iterations=mask_dilate_iterations
        )
        blurred_mask = Image.fromarray(mask_uint8).filter(
            ImageFilter.GaussianBlur(radius=mask_blur_radius)
        )
        blend_mask = np.asarray(blurred_mask).astype(np.float32) / 255.0
        mixed = blend_mask * generated_array + (1 - blend_mask) * source_array
        result = Image.fromarray(
            np.clip(mixed * 255, 0, 255).astype(np.uint8)
        ).convert("RGB")
        return result, mask_mean


def resize_for_diffree(image: Image.Image, resolution: int) -> Image.Image:
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    resized_width = int((width * factor) // 64) * 64
    resized_height = int((height * factor) // 64) * 64
    return ImageOps.fit(
        image,
        (resized_width, resized_height),
        method=Image.Resampling.LANCZOS,
    ).convert("RGB")


def main() -> int:
    args = arguments()
    with args.schedule.open("r", encoding="utf-8") as stream:
        schedule = json.load(stream)

    all_tasks = flatten_tasks(schedule)
    indexed_tasks = list(enumerate(all_tasks))[args.start_index :]
    if args.limit is not None:
        indexed_tasks = indexed_tasks[: args.limit]
    base_seed = int(schedule.get("seed", 2024) if args.seed is None else args.seed)

    print(f"Selected tasks: {len(indexed_tasks)} / {len(all_tasks)}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")

    if args.dry_run:
        for index, task in indexed_tasks:
            objects = objects_for(task)
            print(
                json.dumps(
                    {
                        "index": index,
                        "source": str(source_for(args.data_root, task)),
                        "output": str(destination_for(args.output_root, task)),
                        "objects": objects,
                        "prompts": [prompt_for_object(item) for item in objects],
                        "seed": base_seed + index,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not 0 <= args.gpu_id < torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id={args.gpu_id}; visible GPUs={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.jsonl"
    runner = DiffreeRunner(
        args.config, args.checkpoint, args.vae_checkpoint, device
    )
    succeeded = skipped = failed = 0

    for position, (index, task) in enumerate(indexed_tasks, start=1):
        source_path = source_for(args.data_root, task)
        output_path = destination_for(args.output_root, task)
        task_seed = base_seed + index
        started = time.time()
        object_records: list[dict[str, Any]] = []

        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{position}/{len(indexed_tasks)}] skip {output_path}")
            continue

        try:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as opened:
                original = opened.convert("RGB")
            original_size = original.size
            current = resize_for_diffree(original, args.resolution)
            objects = objects_for(task)
            print(
                f"[{position}/{len(indexed_tasks)}] {source_path} "
                f"objects={objects}"
            )

            for object_index, object_name in enumerate(objects):
                prompt = prompt_for_object(object_name)
                initial_seed = task_seed + object_index
                print(
                    f"  [{object_index + 1}/{len(objects)}] "
                    f"{object_name}: {prompt}"
                )
                for retry in range(args.max_retries + 1):
                    attempt_seed = initial_seed + retry
                    candidate, mask_mean = runner.insert_once(
                        current,
                        prompt,
                        attempt_seed,
                        args.steps,
                        args.text_cfg_scale,
                        args.image_cfg_scale,
                        args.mask_dilate_iterations,
                        args.mask_blur_radius,
                    )
                    if mask_mean >= -0.99:
                        current = candidate
                        object_records.append(
                            {
                                "object_index": object_index,
                                "object": object_name,
                                "prompt": prompt,
                                "seed": attempt_seed,
                                "retries": retry,
                                "mask_mean": round(mask_mean, 6),
                            }
                        )
                        break
                    print(
                        f"    empty mask; retry={retry + 1}, "
                        f"next_seed={attempt_seed + 1}"
                    )
                else:
                    raise RuntimeError(
                        f"No object mask for {object_name!r} after "
                        f"{args.max_retries + 1} attempts"
                    )

            if not args.keep_generated_size and current.size != original_size:
                current = current.resize(original_size, Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            current.save(output_path)
            succeeded += 1
            write_manifest(
                manifest,
                {
                    "status": "ok",
                    "model": "Diffree",
                    "index": index,
                    "seq": task["seq"],
                    "frame_id": task["frame_id"],
                    "source": str(source_path),
                    "output": str(output_path),
                    "class_names": task["class_names"],
                    "objects": object_records,
                    "base_seed": task_seed,
                    "steps": args.steps,
                    "text_cfg_scale": args.text_cfg_scale,
                    "image_cfg_scale": args.image_cfg_scale,
                    "seconds": round(time.time() - started, 3),
                },
            )
        except Exception as error:
            failed += 1
            print(f"  ERROR: {error}", file=sys.stderr)
            write_manifest(
                manifest,
                {
                    "status": "error",
                    "model": "Diffree",
                    "index": index,
                    "seq": task.get("seq"),
                    "frame_id": task.get("frame_id"),
                    "source": str(source_path),
                    "output": str(output_path),
                    "class_names": task.get("class_names"),
                    "objects": object_records,
                    "base_seed": task_seed,
                    "error": repr(error),
                },
            )
            torch.cuda.empty_cache()

    print(f"Done: succeeded={succeeded}, skipped={skipped}, failed={failed}")
    print(f"Manifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
