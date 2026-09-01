#!/usr/bin/env python3
"""Batch Add-it inference for a VisDrone insertion schedule.

Positive class counts are expanded into individual objects. Objects are added
sequentially, so every Add-it call receives the image result from the preceding
insertion. Text prompts do not accumulate: only the object name changes.

Frames are resized without changing their aspect ratio, padded to Add-it's
square canvas with replicated edge pixels, and cropped/restored after the last
insertion.

This script is intended to run from an environment created for the official
NVlabs/Add-it repository. Set ADDIT_ROOT or pass --addit-root when the repo is
not located at /home/qinma/yelo/models/addit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_DATA_ROOT = Path("/home/qinma/yelo/datasets/Visdrone")
DEFAULT_ADDIT_ROOT = Path(
    os.environ.get("ADDIT_ROOT", "/home/qinma/yelo/models/addit")
).expanduser()
DEFAULT_OUTPUT_ROOT = Path("/home/qinma/yelo/outputs/Addit_Visdrone")
DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"

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
    ("motor", "motorcycle"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--addit-root", type=Path, default=DEFAULT_ADDIT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--source-prompt",
        default="An aerial photograph of an urban street scene",
        help="Description shared by the unedited VisDrone frames.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--extended-scale", type=float, default=1.1)
    parser.add_argument("--structure-transfer-step", type=int, default=4)
    parser.add_argument("--blend-steps", type=int, nargs="*", default=[18])
    parser.add_argument(
        "--localization-model",
        choices=(
            "attention",
            "attention_points_sam",
            "attention_box_sam",
            "attention_mask_sam",
            "grounding_sam",
        ),
        default="attention",
    )
    parser.add_argument("--use-offset", action="store_true")
    parser.add_argument("--disable-inversion", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def target_prompt(source_prompt: str, object_name: str) -> str:
    return (
        f"{source_prompt}, with an additional {object_name} visible in the scene, "
        "matching the aerial perspective, scale, lighting, and photographic style"
    )


def write_manifest(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def prepare_for_addit(
    image: Image.Image, resolution: int
) -> tuple[Image.Image, dict[str, int]]:
    """Aspect-preserving resize followed by replicated-edge square padding."""
    source = image.convert("RGB")
    original_width, original_height = source.size
    scale = min(resolution / original_width, resolution / original_height)
    resized_width = max(1, min(resolution, round(original_width * scale)))
    resized_height = max(1, min(resolution, round(original_height * scale)))
    resized = source.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )

    pad_left = (resolution - resized_width) // 2
    pad_right = resolution - resized_width - pad_left
    pad_top = (resolution - resized_height) // 2
    pad_bottom = resolution - resized_height - pad_top

    # First extend the left/right edge columns, then extend the resulting full
    # width's top/bottom rows. This fills the corners without a solid-color seam.
    middle = Image.new("RGB", (resolution, resized_height))
    middle.paste(resized, (pad_left, 0))
    if pad_left:
        left_edge = resized.crop((0, 0, 1, resized_height)).resize(
            (pad_left, resized_height), Image.Resampling.NEAREST
        )
        middle.paste(left_edge, (0, 0))
    if pad_right:
        right_edge = resized.crop(
            (resized_width - 1, 0, resized_width, resized_height)
        ).resize((pad_right, resized_height), Image.Resampling.NEAREST)
        middle.paste(right_edge, (pad_left + resized_width, 0))

    padded = Image.new("RGB", (resolution, resolution))
    padded.paste(middle, (0, pad_top))
    if pad_top:
        top_edge = middle.crop((0, 0, resolution, 1)).resize(
            (resolution, pad_top), Image.Resampling.NEAREST
        )
        padded.paste(top_edge, (0, 0))
    if pad_bottom:
        bottom_edge = middle.crop(
            (0, resized_height - 1, resolution, resized_height)
        ).resize((resolution, pad_bottom), Image.Resampling.NEAREST)
        padded.paste(bottom_edge, (0, pad_top + resized_height))

    geometry = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "canvas_size": resolution,
    }
    return padded, geometry


def restore_from_addit(
    image: Image.Image, geometry: dict[str, int]
) -> Image.Image:
    """Remove preprocessing padding and restore the exact original dimensions."""
    left = geometry["pad_left"]
    top = geometry["pad_top"]
    right = left + geometry["resized_width"]
    bottom = top + geometry["resized_height"]
    cropped = image.convert("RGB").crop((left, top, right, bottom))
    return cropped.resize(
        (geometry["original_width"], geometry["original_height"]),
        Image.Resampling.LANCZOS,
    )


class AdditRunner:
    def __init__(
        self, addit_root: Path, model_name: str, device: Any
    ) -> None:
        import torch

        if not (addit_root / "addit_methods.py").is_file():
            raise FileNotFoundError(
                f"Official Add-it repository not found at {addit_root}. "
                "Set ADDIT_ROOT or pass --addit-root."
            )
        sys.path.insert(0, str(addit_root.resolve()))
        from addit_flux_pipeline import AdditFluxPipeline
        from addit_flux_transformer import AdditFluxTransformer2DModel
        from addit_methods import add_object_real
        from addit_scheduler import AdditFlowMatchEulerDiscreteScheduler

        transformer = AdditFluxTransformer2DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = AdditFluxPipeline.from_pretrained(
            model_name, transformer=transformer, torch_dtype=torch.bfloat16
        ).to(device)
        pipe.scheduler = AdditFlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        self.pipe = pipe
        self.add_object_real = add_object_real

    def insert_once(
        self,
        image: Image.Image,
        source_prompt: str,
        prompt_target: str,
        subject_token: str,
        seed_src: int,
        seed_obj: int,
        args: argparse.Namespace,
    ) -> Image.Image:
        _, edited = self.add_object_real(
            self.pipe,
            source_image=image,
            prompt_source=source_prompt,
            prompt_object=prompt_target,
            subject_token=subject_token,
            seed_src=seed_src,
            seed_obj=seed_obj,
            extended_scale=args.extended_scale,
            structure_transfer_step=args.structure_transfer_step,
            blend_steps=args.blend_steps,
            localization_model=args.localization_model,
            use_offset=args.use_offset,
            show_attention=False,
            use_inversion=not args.disable_inversion,
            display_output=False,
        )
        return edited.convert("RGB")


def main() -> int:
    args = arguments()
    if args.resolution != 1024:
        raise ValueError(
            "The official Add-it real-image implementation is fixed at "
            "1024x1024; use --resolution 1024."
        )
    with args.schedule.open("r", encoding="utf-8") as stream:
        schedule = json.load(stream)

    all_tasks = flatten_tasks(schedule)
    indexed_tasks = list(enumerate(all_tasks))[args.start_index :]
    if args.limit is not None:
        indexed_tasks = indexed_tasks[: args.limit]
    base_seed = int(schedule.get("seed", 2024) if args.seed is None else args.seed)

    print(f"Selected tasks: {len(indexed_tasks)} / {len(all_tasks)}")
    print(f"Add-it root: {args.addit_root}")
    print(f"Model: {args.model}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")

    if args.dry_run:
        for index, task in indexed_tasks:
            source_prompt = args.source_prompt.rstrip(" .")
            insertions = []
            for object_index, object_name in enumerate(objects_for(task)):
                target = target_prompt(source_prompt, object_name)
                insertions.append(
                    {
                        "object": object_name,
                        "source_prompt": source_prompt,
                        "target_prompt": target,
                        "seed_src": base_seed + index,
                        "seed_obj": base_seed + index + object_index,
                    }
                )
            print(
                json.dumps(
                    {
                        "index": index,
                        "source": str(source_for(args.data_root, task)),
                        "output": str(destination_for(args.output_root, task)),
                        "insertions": insertions,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    import torch

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
    runner = AdditRunner(args.addit_root, args.model, device)
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
            current, geometry = prepare_for_addit(original, args.resolution)
            source_prompt = args.source_prompt.rstrip(" .")
            objects = objects_for(task)
            print(
                f"[{position}/{len(indexed_tasks)}] {source_path} objects={objects}"
            )

            for object_index, object_name in enumerate(objects):
                prompt_target = target_prompt(source_prompt, object_name)
                object_seed = task_seed + object_index
                print(
                    f"  [{object_index + 1}/{len(objects)}] {object_name}: "
                    f"{prompt_target}"
                )
                current = runner.insert_once(
                    current,
                    source_prompt,
                    prompt_target,
                    object_name,
                    task_seed,
                    object_seed,
                    args,
                )
                object_records.append(
                    {
                        "status": "inserted",
                        "object_index": object_index,
                        "object": object_name,
                        "source_prompt": source_prompt,
                        "target_prompt": prompt_target,
                        "seed_src": task_seed,
                        "seed_obj": object_seed,
                    }
                )

            current = restore_from_addit(current, geometry)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            current.save(output_path)
            succeeded += 1
            write_manifest(
                manifest,
                {
                    "status": "ok",
                    "model": "Add-it",
                    "index": index,
                    "seq": task["seq"],
                    "frame_id": task["frame_id"],
                    "source": str(source_path),
                    "output": str(output_path),
                    "class_names": task["class_names"],
                    "objects": object_records,
                    "base_seed": task_seed,
                    "extended_scale": args.extended_scale,
                    "structure_transfer_step": args.structure_transfer_step,
                    "blend_steps": args.blend_steps,
                    "localization_model": args.localization_model,
                    "preprocessing": {
                        "method": "aspect_resize_edge_pad",
                        **geometry,
                    },
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
                    "model": "Add-it",
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
