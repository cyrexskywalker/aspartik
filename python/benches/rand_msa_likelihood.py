"""Benchmark CPU and Metal likelihood calculators on random alignments."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood, MetalLikelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.data.msa import MSA
from aspartik.rng import RNG

THREADGROUP_ENV = "ASPARTIK_METAL_THREADS_PER_THREADGROUP"


@dataclass(frozen=True)
class Case:
    sequences: int
    sites: int


@dataclass(frozen=True)
class Result:
    backend: str
    sequences: int
    sites: int
    patterns: int
    mean_us: float
    median_us: float
    stdev_us: float
    likelihood: float
    abs_diff: float | None
    speedup: float | None


def parse_cases(value: str) -> list[Case]:
    cases = []
    for part in value.split(","):
        sequences, sites = part.lower().split("x", maxsplit=1)
        cases.append(Case(int(sequences), int(sites)))
    return cases


def make_msa(case: Case, seed: int) -> MSA:
    names = [str(i) for i in range(case.sequences)]
    return MSA.random(case.sequences, case.sites, names, RNG(seed))


def make_likelihood(kind: str, msa: MSA, seed: int):
    tree = Tree(msa.sequence_names(), RNG(seed))
    kappa = Real(1.0)
    clock_rate = Real(1.0)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)

    likelihood_cls = CPU4Likelihood if kind == "cpu" else MetalLikelihood
    likelihood = likelihood_cls(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
    )
    return tree, likelihood


def measured_likelihood(tree: Tree, likelihood, step: int) -> float:
    # A tiny deterministic scale forces GenericLikelihood to recompute instead
    # of returning the cached value while keeping the tree valid.
    scale = 1.000001 if step % 2 == 0 else 0.999999
    tree.scale(scale)
    value = likelihood.likelihood()
    likelihood.accept()
    tree.accept()
    return value


def time_backend(kind: str, msa: MSA, seed: int, warmup: int, iterations: int):
    tree, likelihood = make_likelihood(kind, msa, seed)

    value = likelihood.likelihood()
    for step in range(warmup):
        value = measured_likelihood(tree, likelihood, step)

    samples = []
    for step in range(iterations):
        start = time.perf_counter_ns()
        value = measured_likelihood(tree, likelihood, warmup + step)
        samples.append((time.perf_counter_ns() - start) / 1_000)

    return likelihood.num_patterns(), value, samples


def summarize(
    backend: str,
    case: Case,
    patterns: int,
    value: float,
    samples: list[float],
    *,
    cpu_mean: float | None = None,
    cpu_value: float | None = None,
) -> Result:
    mean_us = statistics.fmean(samples)
    speedup = None if cpu_mean is None else cpu_mean / mean_us
    abs_diff = None if cpu_value is None else abs(cpu_value - value)
    return Result(
        backend=backend,
        sequences=case.sequences,
        sites=case.sites,
        patterns=patterns,
        mean_us=mean_us,
        median_us=statistics.median(samples),
        stdev_us=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        likelihood=value,
        abs_diff=abs_diff,
        speedup=speedup,
    )


def print_result(result: Result) -> None:
    diff = "-" if result.abs_diff is None else f"{result.abs_diff:.6f}"
    speedup = "-" if result.speedup is None else f"{result.speedup:.2f}x"
    print(
        f"{result.backend:>8} "
        f"{result.sequences:>4}x{result.sites:<7} "
        f"patterns={result.patterns:<7} "
        f"mean={result.mean_us:>10.2f} us "
        f"median={result.median_us:>10.2f} us "
        f"diff={diff:>12} "
        f"speedup={speedup:>8}"
    )


def write_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_svg(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    by_case = {}
    for result in results:
        key = f"{result.sequences}x{result.sites}"
        if key not in by_case:
            cases.append(key)
            by_case[key] = {}
        by_case[key][result.backend] = result.mean_us / 1_000

    max_ms = max(value for case in by_case.values() for value in case.values())
    width = 980
    height = 420
    margin_left = 90
    margin_bottom = 80
    plot_width = width - margin_left - 40
    plot_height = height - 60 - margin_bottom
    group_width = plot_width / len(cases)
    bar_width = min(70, group_width / 3)

    def bar_height(value: float) -> float:
        return value / max_ms * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">CPU vs Metal на rand_msa</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 30}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="55" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        '<text x="18" y="60" font-family="Arial" font-size="13">ms</text>',
    ]

    for i in range(6):
        value = max_ms * i / 5
        y = height - margin_bottom - bar_height(value)
        svg.append(
            f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{width - 30}" y2="{y:.2f}" stroke="#e6e6e6"/>'
        )
        svg.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>'
        )

    for index, case in enumerate(cases):
        center = margin_left + group_width * index + group_width / 2
        cpu_ms = by_case[case].get("CPU")
        metal_ms = by_case[case].get("Metal")
        for offset, backend, color, value in [
            (-bar_width / 1.8, "CPU", "#4e79a7", cpu_ms),
            (bar_width / 1.8, "Metal", "#f28e2b", metal_ms),
        ]:
            if value is None:
                continue
            h = bar_height(value)
            x = center + offset - bar_width / 2
            y = height - margin_bottom - h
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.1f}</text>'
            )
        svg.append(
            f'<text x="{center:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{case}</text>'
        )

    svg.extend(
        [
            '<rect x="390" y="365" width="14" height="14" fill="#4e79a7"/>',
            '<text x="410" y="377" font-family="Arial" font-size="13">CPU</text>',
            '<rect x="470" y="365" width="14" height="14" fill="#f28e2b"/>',
            '<text x="490" y="377" font-family="Arial" font-size="13">Metal</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="32x10000,64x20000,128x50000",
        help="Comma-separated <sequences>x<sites> cases.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--threadgroup", type=int, default=256)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("target/rand_msa_cpu_vs_metal.csv"),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("target/rand_msa_cpu_vs_metal.svg"),
    )
    args = parser.parse_args()

    previous_threadgroup = os.environ.get(THREADGROUP_ENV)
    os.environ[THREADGROUP_ENV] = str(args.threadgroup)

    results = []
    try:
        print(
            " backend case       patterns        mean        median         diff    speedup"
        )
        for case in parse_cases(args.cases):
            msa = make_msa(case, args.seed)

            cpu_patterns, cpu_value, cpu_samples = time_backend(
                "cpu", msa, args.seed, args.warmup, args.iterations
            )
            cpu_result = summarize("CPU", case, cpu_patterns, cpu_value, cpu_samples)
            results.append(cpu_result)
            print_result(cpu_result)

            metal_patterns, metal_value, metal_samples = time_backend(
                "metal", msa, args.seed, args.warmup, args.iterations
            )
            metal_result = summarize(
                "Metal",
                case,
                metal_patterns,
                metal_value,
                metal_samples,
                cpu_mean=cpu_result.mean_us,
                cpu_value=cpu_value,
            )
            results.append(metal_result)
            print_result(metal_result)
    finally:
        if previous_threadgroup is None:
            os.environ.pop(THREADGROUP_ENV, None)
        else:
            os.environ[THREADGROUP_ENV] = previous_threadgroup

    write_csv(args.csv, results)
    write_svg(args.svg, results)
    print(f"CSV: {args.csv}")
    print(f"SVG: {args.svg}")


if __name__ == "__main__":
    main()
