#!/usr/bin/env python3
"""Batch FLUX.2-dev inference for the UAVDT insertion schedule.

The JSON has no separate mask path, so each input image is assumed to already
contain the masked regions referred to by the editing prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2Pipeline
from PIL import Image


OLD_ROOT_MARKER = "/datasets/UAVDT/"
DEFAULT_DATA_ROOT = Path("/home/qinma/yelo/datasets/UAVDT")
DEFAULT_MODEL_PATH = Path("/home/qinma/yelo/models/FLUX.2-dev")
DEFAULT_OUTPUT_ROOT = Path("/home/qinma/yelo/outputs/FLUX.2-dev_UAVDT")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-generated-size", action="store_true")
    return parser.parse_args()


def flatten_tasks(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for tasks in schedule["sequences"].values() for task in tasks]


def resolve_source(path: str, data_root: Path) -> Path:
    path = path.replace("\\", "/")
    if OLD_ROOT_MARKER not in path:
        raise ValueError(f"Cannot replace UAVDT root in path: {path}")
    relative = path.split(OLD_ROOT_MARKER, 1)[1]
    return data_root / relative


def object_phrase(counts: dict[str, int]) -> str:
    names = (("car", "cars"), ("truck", "trucks"), ("bus", "buses"))
    parts: list[str] = []
    for singular, plural in names:
        count = int(counts.get(singular, 0))
        if count > 0:
            parts.append(f"{count} {singular if count == 1 else plural}")
    if not parts:
        raise ValueError(f"No objects requested: {counts}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def prompt_for(task: dict[str, Any]) -> str:
    objects = object_phrase(task["class_names"])
    return (
        f"Preseve all exisitng objects and keep the rest of image unchanged. Insert {objects} in aerial view. Match the aerial perspective, lighting, and style of the scene. "
    )


def destination_for(root: Path, task: dict[str, Any]) -> Path:
    return root / task["seq"] / f"img{int(task['frame_id']):06d}.png"


def write_manifest(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def load_pipeline(model_path: Path, gpu_id: int) -> Flux2Pipeline:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    pipeline = Flux2Pipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipeline


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
    print(f"Model: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")

    if args.dry_run:
        for index, task in indexed_tasks:
            print(
                json.dumps(
                    {
                        "index": index,
                        "source": str(resolve_source(task["image"], args.data_root)),
                        "output": str(destination_for(args.output_root, task)),
                        "prompt": prompt_for(task),
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
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.jsonl"
    pipeline = load_pipeline(args.model_path, args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    succeeded = skipped = failed = 0

    for position, (index, task) in enumerate(indexed_tasks, start=1):
        source_path = resolve_source(task["image"], args.data_root)
        output_path = destination_for(args.output_root, task)
        prompt = prompt_for(task)
        seed = base_seed + index

        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{position}/{len(indexed_tasks)}] skip {output_path}")
            continue

        started = time.time()
        try:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
            source_size = source.size
            print(f"[{position}/{len(indexed_tasks)}] {source_path}")
            print(f"  {prompt}")

            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                result = pipeline(
                    image=source,
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                ).images[0]

            if not args.keep_generated_size and result.size != source_size:
                result = result.resize(source_size, Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            succeeded += 1
            write_manifest(
                manifest,
                {
                    "status": "ok",
                    "index": index,
                    "seq": task["seq"],
                    "frame_id": task["frame_id"],
                    "source": str(source_path),
                    "output": str(output_path),
                    "class_names": task["class_names"],
                    "prompt": prompt,
                    "seed": seed,
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
                    "index": index,
                    "seq": task.get("seq"),
                    "frame_id": task.get("frame_id"),
                    "source": str(source_path),
                    "output": str(output_path),
                    "prompt": prompt,
                    "seed": seed,
                    "error": repr(error),
                },
            )
            torch.cuda.empty_cache()

    print(f"Done: succeeded={succeeded}, skipped={skipped}, failed={failed}")
    print(f"Manifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
