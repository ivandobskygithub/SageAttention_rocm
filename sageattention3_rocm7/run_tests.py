#!/usr/bin/env python
"""
Test runner for SageAttention3 ROCm port.

This script runs the test suite and generates a summary report.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(
    test_dir: str = "tests",
    markers: str = None,
    verbose: bool = True,
    coverage: bool = False,
    report_file: str = None,
):
    """Run pytest with specified options"""

    cmd = [sys.executable, "-m", "pytest", test_dir]

    if verbose:
        cmd.append("-v")

    if markers:
        cmd.extend(["-m", markers])

    if coverage:
        cmd.extend([
            "--cov=sage_attention_rocm7",
            "--cov-report=html",
            "--cov-report=term",
        ])

    if report_file:
        cmd.extend(["--html", report_file, "--self-contained-html"])

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run SageAttention3 ROCm tests")

    parser.add_argument(
        "--dir",
        type=str,
        default="tests",
        help="Test directory (default: tests)"
    )

    parser.add_argument(
        "-m", "--markers",
        type=str,
        help="Run tests with specific markers (e.g., 'gpu', 'accuracy', 'not slow')"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report (requires pytest-cov)"
    )

    parser.add_argument(
        "--report",
        type=str,
        help="Generate HTML report (requires pytest-html)"
    )

    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Run only GPU tests"
    )

    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Run only accuracy tests"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (exclude slow tests)"
    )

    args = parser.parse_args()

    # Build markers string
    markers = args.markers
    if args.gpu_only:
        markers = "gpu"
    elif args.accuracy_only:
        markers = "accuracy"
    elif args.quick:
        markers = "not slow"

    # Run tests
    return_code = run_tests(
        test_dir=args.dir,
        markers=markers,
        verbose=args.verbose,
        coverage=args.coverage,
        report_file=args.report,
    )

    sys.exit(return_code)


if __name__ == "__main__":
    main()
