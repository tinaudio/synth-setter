import os
import time
from collections.abc import Callable

import psutil
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.kosc_datamodule import make_sig
from src.data.ksin_datamodule import make_sin


def test_fn_performance(fn: Callable):
    k_values = [1, 4, 8, 16, 32]
    length = 1024

    # Store the results in a dict
    results = {}

    process = psutil.Process(os.getpid())

    for k in k_values:
        # Create the input tensor of shape (1, 3*k) with uniform samples in [-1,1]
        params = 2.0 * torch.rand((1, 3 * k)) - 1.0

        # burn in
        _ = fn(params, length, break_symmetry=False)

        # Measure memory before
        mem_before = process.memory_info().rss

        # Measure time before
        t_start = time.perf_counter()

        # Call the function
        for _ in range(1):
            _ = fn(params, length, break_symmetry=False)

        # Measure time after
        t_end = time.perf_counter()

        # Measure memory after
        mem_after = process.memory_info().rss

        # Calculate difference
        runtime = t_end - t_start  # in seconds
        mem_usage = mem_after - mem_before  # in bytes

        results[k] = {
            "runtime_seconds": runtime,
            "memory_bytes": mem_usage,
        }

    # Print out results
    print("Performance Results:")
    for k, result in results.items():
        print(
            f"k={k} => runtime: {result['runtime_seconds']:.6f}s, "
            f"memory: {result['memory_bytes']} bytes"
        )


if __name__ == "__main__":
    print("kosc")
    test_fn_performance(make_sig)
    print("ksin")
    test_fn_performance(make_sin)
