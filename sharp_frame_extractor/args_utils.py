import argparse
import os


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
