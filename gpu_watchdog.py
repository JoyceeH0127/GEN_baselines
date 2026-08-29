#!/usr/bin/env python3
"""Reserve genuinely idle NVIDIA GPUs and yield to real workloads.

The daemon observes every GPU for OBSERVE_SECONDS. A GPU is considered idle only
when it has no external compute process, low utilisation, and no meaningful
memory/utilisation change throughout that window. It then starts one child holder
which allocates a configurable fraction of the currently free VRAM.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


LOG = logging.getLogger("gpu-watchdog")
STOP = False


def stop_handler(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def run_smi(args: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *args], text=True, capture_output=True, timeout=15
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    return result.stdout


def gpu_samples() -> dict[str, tuple[int, int]]:
    output = run_smi([
        "--query-gpu=uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ])
    samples: dict[str, tuple[int, int]] = {}
    for line in output.splitlines():
        uuid, util, used = (part.strip() for part in line.split(","))
        samples[uuid] = (int(util), int(used))
    return samples


def compute_processes() -> dict[str, set[int]]:
    output = run_smi([
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ])
    result: dict[str, set[int]] = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or fields[1] in {"", "[Not Supported]", "N/A"}:
            continue
        try:
            result.setdefault(fields[0], set()).add(int(fields[1]))
        except ValueError:
            continue
    return result


def gpu_order() -> list[str]:
    output = run_smi(["--query-gpu=uuid", "--format=csv,noheader,nounits"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def external_pids(uuid: str, holders: dict[str, subprocess.Popen[bytes]]) -> set[int]:
    owned = {proc.pid for proc in holders.values() if proc.poll() is None}
    return compute_processes().get(uuid, set()) - owned


def load_cudart() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise RuntimeError("CUDA Runtime (libcudart.so) was not found")


def check_cuda(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {code}")


def holder(device_index: int, fraction: float, headroom_mib: int) -> int:
    cudart = load_cudart()
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    check_cuda(cudart.cudaSetDevice(device_index), "cudaSetDevice")
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    check_cuda(cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
    headroom = headroom_mib * 1024 * 1024
    amount = int(max(0, free.value - headroom) * fraction)
    if amount <= 0:
        raise RuntimeError("not enough free VRAM after configured headroom")
    pointer = ctypes.c_void_p()
    check_cuda(cudart.cudaMalloc(ctypes.byref(pointer), amount), "cudaMalloc")
    print(f"holder allocated {amount // (1024 * 1024)} MiB on GPU {device_index}", flush=True)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        while not STOP:
            time.sleep(1)
    finally:
        cudart.cudaFree(pointer)
    return 0


def observe_idle_gpus(
    uuids: list[str],
    holders: dict[str, subprocess.Popen[bytes]],
    seconds: int,
    sample_seconds: int,
    max_util: int,
    max_memory_change: int,
) -> set[str]:
    readings: dict[str, list[tuple[int, int]]] = {uuid: [] for uuid in uuids}
    candidates = set(uuids)
    deadline = time.monotonic() + seconds
    while not STOP and time.monotonic() < deadline:
        owned = {proc.pid for proc in holders.values() if proc.poll() is None}
        processes = compute_processes()
        samples = gpu_samples()
        for uuid in list(candidates):
            external = processes.get(uuid, set()) - owned
            sample = samples.get(uuid)
            if external or sample is None or sample[0] > max_util:
                candidates.discard(uuid)
            else:
                readings[uuid].append(sample)
        time.sleep(min(sample_seconds, max(0, deadline - time.monotonic())))
    if STOP:
        return set()
    idle: set[str] = set()
    for uuid in candidates:
        if not readings[uuid]:
            continue
        utils = [item[0] for item in readings[uuid]]
        memory = [item[1] for item in readings[uuid]]
        if max(utils) <= max_util and max(memory) - min(memory) <= max_memory_change:
            idle.add(uuid)
    return idle


def terminate(proc: subprocess.Popen[bytes], timeout: int = 15) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def daemon(args: argparse.Namespace) -> int:
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOG.error("another gpu-watchdog instance is already running")
        return 1
    lock.write(str(os.getpid()))
    lock.flush()

    uuids = gpu_order()
    if not uuids:
        raise RuntimeError("no NVIDIA GPU found")
    holders: dict[str, subprocess.Popen[bytes]] = {}
    next_scan = 0.0
    LOG.info("watching %d GPU(s)", len(uuids))
    try:
        while not STOP:
            # Yield reservations promptly when a real CUDA process appears.
            for uuid, proc in list(holders.items()):
                if proc.poll() is not None:
                    LOG.warning("holder for %s exited with code %s", uuid, proc.returncode)
                    del holders[uuid]
                elif external_pids(uuid, holders):
                    LOG.info("external workload detected on %s; releasing reservation", uuid)
                    terminate(proc)
                    del holders[uuid]

            if time.monotonic() >= next_scan:
                scan_started = time.monotonic()
                unreserved = [uuid for uuid in uuids if uuid not in holders]
                if unreserved:
                    LOG.info("observing %d GPU(s) in parallel for %ds", len(unreserved), args.observe_seconds)
                    idle = observe_idle_gpus(
                        unreserved, holders, args.observe_seconds, args.sample_seconds,
                        args.max_util, args.max_memory_change_mib,
                    )
                else:
                    idle = set()
                for index, uuid in enumerate(uuids):
                    if STOP:
                        break
                    if uuid in idle:
                        command = [
                            sys.executable, str(Path(__file__).resolve()), "--holder",
                            "--device", str(index), "--fraction", str(args.fraction),
                            "--headroom-mib", str(args.headroom_mib),
                        ]
                        proc = subprocess.Popen(command)
                        time.sleep(2)
                        if proc.poll() is None:
                            holders[uuid] = proc
                            LOG.info("reserved idle GPU %s (holder pid %d)", uuid, proc.pid)
                        else:
                            LOG.error("could not reserve %s; holder exited %s", uuid, proc.returncode)
                # Keep scan starts on the requested cadence; observation time is
                # part of the interval rather than added on top of it.
                next_scan = scan_started + args.interval_seconds
            time.sleep(args.active_poll_seconds)
    finally:
        for proc in holders.values():
            terminate(proc)
    return 0


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=positive, default=1800)
    parser.add_argument("--observe-seconds", type=positive, default=300)
    parser.add_argument("--sample-seconds", type=positive, default=15)
    parser.add_argument("--active-poll-seconds", type=positive, default=5)
    parser.add_argument("--max-util", type=int, default=2, help="idle utilization ceiling (%%)")
    parser.add_argument("--max-memory-change-mib", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=0.90)
    parser.add_argument("--headroom-mib", type=int, default=1024)
    parser.add_argument("--lock-file", default="/tmp/gpu-watchdog.lock")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--holder", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.max_util < 0 or args.max_memory_change_mib < 0:
        parser.error("thresholds cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    if args.holder:
        if args.device is None:
            raise RuntimeError("--device is required in holder mode")
        return holder(args.device, args.fraction, args.headroom_mib)
    return daemon(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        LOG.error("%s", exc)
        raise SystemExit(1)
