"""
Compare CPU and Metal likelihood evaluation time on the same model state.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood, MetalLikelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

THREADGROUP_ENV = "ASPARTIK_METAL_THREADS_PER_THREADGROUP"


def time_calls(fn: Callable[[], float], iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / 1_000)
    return samples


def summarize(name: str, samples: list[float], value: float) -> None:
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(
        f"{name:>8}: "
        f"mean={mean:10.2f} us  "
        f"median={median:10.2f} us  "
        f"stdev={stdev:10.2f} us  "
        f"likelihood={value:.6f}"
    )


def make_cpu_likelihood(fasta_path: str):
    msa = read_msa_from_fasta(fasta_path)
    cpu_tree = Tree(msa.sequence_names(), RNG(4))

    kappa = Real(1.0)
    clock_rate = Real(1.0)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)

    cpu = CPU4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=cpu_tree,
    )

    return cpu


def make_metal_likelihood(fasta_path: str):
    msa = read_msa_from_fasta(fasta_path)
    metal_tree = Tree(msa.sequence_names(), RNG(4))

    kappa = Real(1.0)
    clock_rate = Real(1.0)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)

    metal = MetalLikelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=metal_tree,
    )

    return metal


def parse_threadgroups(value: str | None) -> list[int | None]:
    if value is None:
        return [None]

    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta",
        nargs="?",
        default="data/alignments/H1N1pdm_2009.fasta",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--threadgroups",
        help="Comma-separated Metal threadgroup sizes, for example 64,128,256",
    )
    args = parser.parse_args()

    cpu = make_cpu_likelihood(args.fasta)

    for _ in range(args.warmup):
        cpu.likelihood()

    cpu_samples = time_calls(cpu.likelihood, args.iterations)
    cpu_value = cpu.likelihood()
    summarize("CPU", cpu_samples, cpu_value)

    previous_threadgroup = os.environ.get(THREADGROUP_ENV)
    try:
        for threadgroup in parse_threadgroups(args.threadgroups):
            if threadgroup is None:
                os.environ.pop(THREADGROUP_ENV, None)
                label = "Metal"
            else:
                os.environ[THREADGROUP_ENV] = str(threadgroup)
                label = f"Metal/{threadgroup}"

            metal = make_metal_likelihood(args.fasta)
            for _ in range(args.warmup):
                metal.likelihood()

            metal_samples = time_calls(metal.likelihood, args.iterations)
            metal_value = metal.likelihood()
            summarize(label, metal_samples, metal_value)
            print(f"{label:>8}: abs diff={abs(cpu_value - metal_value):.6f}")
    finally:
        if previous_threadgroup is None:
            os.environ.pop(THREADGROUP_ENV, None)
        else:
            os.environ[THREADGROUP_ENV] = previous_threadgroup


if __name__ == "__main__":
    main()
