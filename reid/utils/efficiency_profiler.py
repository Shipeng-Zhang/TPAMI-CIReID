import csv
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


class EfficiencyProfiler:
    def __init__(self):
        self.enabled = False
        self.warmup = 10
        self.max_samples = 100
        self.calls = defaultdict(int)
        self.records = defaultdict(list)

    def configure(self, enabled=False, warmup=10, max_samples=100):
        self.enabled, self.warmup, self.max_samples = enabled, warmup, max_samples
        print("Efficiency profiler: {} (warmup={}, samples={})".format(
            "enabled" if enabled else "disabled", warmup, max_samples))

    @contextmanager
    def measure(self, name):
        call = self.calls[name]
        self.calls[name] += 1
        warmup = 0 if name == "dynamic_prototype_update" else self.warmup
        active = self.enabled and call >= warmup and len(self.records[name]) < self.max_samples
        if not active:
            yield
            return
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            allocated0, reserved0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            start.record()
        wall0 = time.perf_counter()
        yield
        if cuda:
            end.record(); torch.cuda.synchronize()
            gpu_ms = start.elapsed_time(end)
            allocated1, reserved1 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        else:
            gpu_ms = 0.; allocated0 = allocated1 = reserved0 = reserved1 = 0
            peak_allocated = peak_reserved = 0
        self.records[name].append({
            "wall_ms": (time.perf_counter() - wall0) * 1000, "gpu_ms": gpu_ms,
            "allocated_delta_mb": (allocated1 - allocated0) / 2 ** 20,
            "reserved_delta_mb": (reserved1 - reserved0) / 2 ** 20,
            "peak_allocated_mb": peak_allocated / 2 ** 20,
            "peak_reserved_mb": peak_reserved / 2 ** 20})

    def save(self, path):
        if not self.enabled or not self.records: return
        fields = ["module", "samples", "wall_mean_ms", "wall_std_ms", "gpu_mean_ms", "gpu_std_ms",
                  "allocated_delta_mean_mb", "reserved_delta_mean_mb", "peak_allocated_mb",
                  "peak_reserved_mb", "time_share_percent"]
        overall = np.mean([r["wall_ms"] for r in self.records["overall_framework"]])
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
            for name, records in self.records.items():
                wall = np.array([r["wall_ms"] for r in records]); gpu = np.array([r["gpu_ms"] for r in records])
                writer.writerow({"module": name, "samples": len(records), "wall_mean_ms": wall.mean(),
                    "wall_std_ms": wall.std(), "gpu_mean_ms": gpu.mean(), "gpu_std_ms": gpu.std(),
                    "allocated_delta_mean_mb": np.mean([r["allocated_delta_mb"] for r in records]),
                    "reserved_delta_mean_mb": np.mean([r["reserved_delta_mb"] for r in records]),
                    "peak_allocated_mb": max(r["peak_allocated_mb"] for r in records),
                    "peak_reserved_mb": max(r["peak_reserved_mb"] for r in records),
                    "time_share_percent": ("" if name == "dynamic_prototype_update" else
                        (100 * wall.mean() / overall if name != "overall_framework" else 100.))})
        print("Efficiency profile saved to {}".format(path))


profiler = EfficiencyProfiler()
