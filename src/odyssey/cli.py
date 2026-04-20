"""Command-line interface for Odyssey."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from odyssey.experiments.registry import (
    export_paper_assets,
    generate_synthetic_data,
    make_figures,
    run_ablation_suite,
    run_all_experiments,
    run_baseline_suite,
    run_odyssey_experiment,
)
from odyssey.quantum.workflows import (
    run_quantum_algorithms_workflow,
    run_quantum_foundations_workflow,
    run_quantum_suite_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Odyssey research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-synthetic", help="Generate the synthetic benchmark")
    gen.add_argument("--config", default="configs/synthetic_small.yaml")
    gen.add_argument("--output", default=None)

    baselines = subparsers.add_parser("run-baselines", help="Run the baseline suite")
    baselines.add_argument("--config", default="configs/baseline_suite.yaml")

    odyssey = subparsers.add_parser("run-odyssey", help="Run Odyssey-Risk")
    odyssey.add_argument("--config", default="configs/synthetic_research.yaml")

    run_all = subparsers.add_parser("run-all", help="Run baselines and Odyssey together")
    run_all.add_argument("--config", default="configs/synthetic_small.yaml")

    ablations = subparsers.add_parser("run-ablations", help="Run ablation experiments")
    ablations.add_argument("--config", default="configs/ablation_suite.yaml")

    figures = subparsers.add_parser("make-figures", help="Regenerate figures from saved outputs")
    figures.add_argument("--report", default="outputs/reports/latest_metrics.json")

    paper = subparsers.add_parser("export-paper-assets", help="Export tables and figure manifests for paper support")
    paper.add_argument("--source", default="outputs")

    quantum_foundations = subparsers.add_parser(
        "quantum-foundations",
        help="Run Odyssey's deterministic quantum foundations track",
    )
    quantum_foundations.add_argument("--config", default="configs/quantum_foundations.yaml")

    quantum_algorithms = subparsers.add_parser(
        "quantum-algorithms",
        help="Run Odyssey's educational quantum algorithm suite",
    )
    quantum_algorithms.add_argument("--config", default="configs/quantum_algorithms.yaml")

    quantum_suite = subparsers.add_parser(
        "quantum-suite",
        help="Run the full Odyssey quantum track (foundations + algorithms)",
    )
    quantum_suite.add_argument("--config", default="configs/quantum_suite.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-synthetic":
        path = generate_synthetic_data(args.config, output_path=args.output)
        print(path)
    elif args.command == "run-baselines":
        print(run_baseline_suite(args.config))
    elif args.command == "run-odyssey":
        print(run_odyssey_experiment(args.config))
    elif args.command == "run-all":
        print(run_all_experiments(args.config))
    elif args.command == "run-ablations":
        print(run_ablation_suite(args.config))
    elif args.command == "make-figures":
        print(make_figures(Path(args.report)))
    elif args.command == "export-paper-assets":
        print(export_paper_assets(Path(args.source)))
    elif args.command == "quantum-foundations":
        print(run_quantum_foundations_workflow(args.config))
    elif args.command == "quantum-algorithms":
        print(run_quantum_algorithms_workflow(args.config))
    elif args.command == "quantum-suite":
        print(run_quantum_suite_workflow(args.config))


if __name__ == "__main__":
    main()

