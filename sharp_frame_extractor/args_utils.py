import argparse
import os

import psutil

MIN_MEMORY_LIMIT = 4096


def positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be an integer") from e
    if n <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return n


def positive_float(value: str) -> float:
    try:
        x = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be a number") from e
    if x <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def default_concurrency() -> tuple[int, int]:
    cpu = os.cpu_count() or 1

    if cpu <= 2:
        jobs = 1
    elif cpu <= 4:
        jobs = 2
    elif cpu <= 8:
        jobs = 3
    else:
        jobs = 4

    workers = max(1, int(cpu * 0.8))
    return jobs, workers


def default_memory_limit_mb(safe_factor: float = 0.8) -> int:
    memory_info = psutil.virtual_memory()

    total_bytes = 0

    try:
        total_bytes = memory_info.total
    except AttributeError:
        pass

    # Fallback to 4GB if detection failed or 0
    if total_bytes <= 0:
        return MIN_MEMORY_LIMIT

    # Return n% of total memory in MB
    return int((total_bytes * 0.8) / (1024 * 1024))
