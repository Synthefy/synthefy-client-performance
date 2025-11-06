#!/usr/bin/env python3
"""
Performance test script for Synthefy Forecasting API.

This script benchmarks forecasting performance across GPU and CPU endpoints
with various parameter combinations, measuring latency statistics.
"""

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

from synthefy import SynthefyAPIClient
from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


def generate_test_data(
    history_length: int, forecast_length: int, num_scenarios: int
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Generate test data for performance testing.

    Parameters
    ----------
    history_length : int
        Number of historical data points
    forecast_length : int
        Number of forecast points
    num_scenarios : int
        Number of scenarios to generate

    Returns
    -------
    Tuple[List[pd.DataFrame], List[pd.DataFrame]]
        Lists of history and target DataFrames, one per scenario
    """
    history_dfs = []
    target_dfs = []

    # Base date for historical data
    base_date = pd.Timestamp("2023-01-01")

    for scenario_idx in range(num_scenarios):
        # Generate historical timestamps
        history_dates = pd.date_range(
            base_date, periods=history_length, freq="D"
        )

        # Generate synthetic sales data with trend and some variation
        # Each scenario has slightly different base values
        base_value = 100.0 + scenario_idx * 10
        trend = np.linspace(0, 20, history_length)
        noise = np.random.normal(0, 5, history_length)
        history_values = base_value + trend + noise

        history_data = {
            "date": history_dates,
            "sales": history_values,
        }
        history_df = pd.DataFrame(history_data)
        history_dfs.append(history_df)

        # Generate target timestamps (immediately after history)
        target_start = history_dates[-1] + pd.Timedelta(days=1)
        target_dates = pd.date_range(
            target_start, periods=forecast_length, freq="D"
        )

        # Target data with NaN values to forecast
        target_data = {
            "date": target_dates,
            "sales": [np.nan] * forecast_length,
        }
        target_df = pd.DataFrame(target_data)
        target_dfs.append(target_df)

    return history_dfs, target_dfs


def run_single_forecast(
    client: SynthefyAPIClient,
    history_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    model: str,
) -> float:
    """
    Run a single forecast request and measure latency.

    Parameters
    ----------
    client : SynthefyAPIClient
        The API client to use
    history_dfs : List[pd.DataFrame]
        List of history DataFrames
    target_dfs : List[pd.DataFrame]
        List of target DataFrames
    model : str
        Model name to use

    Returns
    -------
    float
        Latency in seconds
    """
    start_time = time.perf_counter()
    client.forecast_dfs(
        history_dfs=history_dfs,
        target_dfs=target_dfs,
        target_col="sales",
        timestamp_col="date",
        metadata_cols=[],
        leak_cols=[],
        model=model,
    )
    end_time = time.perf_counter()
    return end_time - start_time


def run_performance_test(
    base_url: str,
    model: str,
    history_length: int,
    forecast_length: int,
    num_scenarios: int,
    num_warmup: int = 5,
    num_runs: int = 15,
    api_key: str = "",
) -> Tuple[List[float], bool]:
    """
    Run performance test for a specific configuration.

    Parameters
    ----------
    base_url : str
        Base URL of the API endpoint
    model : str
        Model name to use
    history_length : int
        Number of historical data points
    forecast_length : int
        Number of forecast points
    num_scenarios : int
        Number of scenarios
    num_warmup : int, default 5
        Number of warmup runs to discard
    num_runs : int, default 15
        Total number of runs
    api_key : str, optional
        API key for authentication

    Returns
    -------
    Tuple[List[float], bool]
        List of latencies (excluding warmup) and success status
    """
    # Generate test data
    history_dfs, target_dfs = generate_test_data(
        history_length, forecast_length, num_scenarios
    )

    latencies = []
    success = False

    try:
        with SynthefyAPIClient(api_key=api_key, base_url=base_url) as client:
            # Run all iterations
            for i in range(num_runs):
                try:
                    latency = run_single_forecast(
                        client, history_dfs, target_dfs, model
                    )
                    # Only collect latencies after warmup
                    if i >= num_warmup:
                        latencies.append(latency)
                    success = True
                except Exception as e:
                    print(
                        f"  ⚠️  Run {i + 1} failed: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    # Continue with next run
                    if i >= num_warmup:
                        # If we're past warmup, we still want to record this as a failed run
                        # but we'll use a very high latency value to indicate failure
                        latencies.append(float("inf"))

    except APIConnectionError as e:
        print(f"  ❌ Connection Error: {e}", file=sys.stderr)
    except Exception as e:
        print(
            f"  ❌ Unexpected Error: {type(e).__name__}: {e}", file=sys.stderr
        )

    return latencies, success


def calculate_statistics(latencies: List[float]) -> dict:
    """
    Calculate performance statistics from latency measurements.

    Parameters
    ----------
    latencies : List[float]
        List of latency measurements

    Returns
    -------
    dict
        Dictionary containing mean, median, p95, std_dev, and total_time
    """
    if not latencies:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "std_dev": None,
            "total": None,
        }

    # Filter out infinite values (failed runs)
    valid_latencies = [
        latency for latency in latencies if latency != float("inf")
    ]

    if not valid_latencies:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "std_dev": None,
            "total": None,
        }

    stats = {
        "mean": np.mean(valid_latencies),
        "median": np.median(valid_latencies),
        "std_dev": np.std(valid_latencies),
        "total": np.sum(valid_latencies),
    }

    # Calculate p95 (95th percentile)
    if len(valid_latencies) > 0:
        stats["p95"] = np.percentile(valid_latencies, 95)
    else:
        stats["p95"] = None

    return stats


def format_statistics(stats: dict) -> dict:
    """
    Format statistics for display.

    Parameters
    ----------
    stats : dict
        Statistics dictionary

    Returns
    -------
    dict
        Formatted statistics with string values
    """
    formatted = {}
    for key, value in stats.items():
        if value is None:
            formatted[key] = "N/A"
        elif value == float("inf"):
            formatted[key] = "INF"
        else:
            formatted[key] = f"{value:.4f}"
    return formatted


def print_single_result(result: dict, is_first: bool = False):
    """
    Print a single result row in real-time.

    Parameters
    ----------
    result : dict
        Result dictionary
    is_first : bool, default False
        Whether this is the first result (to print header)
    """
    header = (
        "Device | Forecast Length | Scenarios | Mean (s) | "
        "Median (s) | p95 (s) | Std Dev (s) | Total (s)"
    )

    if is_first:
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

    device = result["device"]
    forecast_length = result["forecast_length"]
    num_scenarios = result["num_scenarios"]
    stats = format_statistics(result["stats"])

    row = (
        f"{device:6s} | {forecast_length:15d} | {num_scenarios:9d} | "
        f"{stats['mean']:8s} | {stats['median']:9s} | {stats['p95']:6s} | "
        f"{stats['std_dev']:10s} | {stats['total']:8s}"
    )
    print(row)


def print_results_table(results: List[dict]):
    """
    Print formatted results table.

    Parameters
    ----------
    results : List[dict]
        List of result dictionaries
    """
    # Header
    header = (
        "Device | Forecast Length | Scenarios | Mean (s) | "
        "Median (s) | p95 (s) | Std Dev (s) | Total (s)"
    )
    print("\n" + "=" * len(header))
    print("SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Results
    for result in results:
        device = result["device"]
        forecast_length = result["forecast_length"]
        num_scenarios = result["num_scenarios"]
        stats = format_statistics(result["stats"])

        row = (
            f"{device:6s} | {forecast_length:15d} | {num_scenarios:9d} | "
            f"{stats['mean']:8s} | {stats['median']:9s} | {stats['p95']:6s} | "
            f"{stats['std_dev']:10s} | {stats['total']:8s}"
        )
        print(row)

    print("=" * len(header))


def get_csv_fieldnames() -> List[str]:
    """Get the CSV field names."""
    return [
        "device",
        "forecast_length",
        "num_scenarios",
        "mean",
        "median",
        "p95",
        "std_dev",
        "total",
        "success",
    ]


def initialize_csv_file(output_file: str):
    """
    Initialize CSV file with headers if it doesn't exist.

    Parameters
    ----------
    output_file : str
        Path to the output CSV file
    """
    if not os.path.exists(output_file):
        fieldnames = get_csv_fieldnames()
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


def append_result_to_csv(result: dict, output_file: str):
    """
    Append a single result to the CSV file.

    Parameters
    ----------
    result : dict
        Result dictionary containing performance statistics
    output_file : str
        Path to the output CSV file
    """
    fieldnames = get_csv_fieldnames()
    stats = result["stats"]
    row = {
        "device": result["device"],
        "forecast_length": result["forecast_length"],
        "num_scenarios": result["num_scenarios"],
        "mean": stats.get("mean") if stats.get("mean") is not None else "",
        "median": stats.get("median")
        if stats.get("median") is not None
        else "",
        "p95": stats.get("p95") if stats.get("p95") is not None else "",
        "std_dev": stats.get("std_dev")
        if stats.get("std_dev") is not None
        else "",
        "total": stats.get("total") if stats.get("total") is not None else "",
        "success": result["success"],
    }

    with open(output_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)
        csvfile.flush()  # Ensure data is written immediately


def save_results_to_csv(results: List[dict], output_file: str):
    """
    Save performance test results to a CSV file (for backward compatibility).

    Parameters
    ----------
    results : List[dict]
        List of result dictionaries containing performance statistics
    output_file : str
        Path to the output CSV file
    """
    fieldnames = get_csv_fieldnames()

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            stats = result["stats"]
            row = {
                "device": result["device"],
                "forecast_length": result["forecast_length"],
                "num_scenarios": result["num_scenarios"],
                "mean": stats.get("mean")
                if stats.get("mean") is not None
                else "",
                "median": stats.get("median")
                if stats.get("median") is not None
                else "",
                "p95": stats.get("p95") if stats.get("p95") is not None else "",
                "std_dev": stats.get("std_dev")
                if stats.get("std_dev") is not None
                else "",
                "total": stats.get("total")
                if stats.get("total") is not None
                else "",
                "success": result["success"],
            }
            writer.writerow(row)

    print(f"\nResults saved to CSV: {output_file}")


def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(
        description="Performance test for Synthefy Forecasting API"
    )
    parser.add_argument(
        "--gpu-url",
        type=str,
        default="http://localhost:9001",
        help="GPU endpoint base URL (default: http://localhost:9001)",
    )
    parser.add_argument(
        "--cpu-url",
        type=str,
        default="http://localhost:9002",
        help="CPU endpoint base URL (default: http://localhost:9002)",
    )
    parser.add_argument(
        "--gpu-model",
        type=str,
        default="sfm-moe-v1",
        help="Model to use for GPU endpoint (default: sfm-moe-v1)",
    )
    parser.add_argument(
        "--cpu-model",
        type=str,
        default="sfm-moe-v1",
        help="Model to use for CPU endpoint (default: prophet)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (default: None)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup runs to discard (default: 5)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=15,
        help="Total number of runs per test (default: 15)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to CSV file to save results (default: None, no CSV output)",
    )

    args = parser.parse_args()

    # Configuration
    devices = [
        {"name": "GPU", "url": args.gpu_url, "model": args.gpu_model},
        {"name": "CPU", "url": args.cpu_url, "model": args.cpu_model},
    ]

    forecast_lengths = [10, 100, 1000]
    num_scenarios_list = [1, 2, 4, 8, 16]

    print("Synthefy Forecasting API Performance Test")
    print("=" * 60)
    print(f"GPU Endpoint: {args.gpu_url} (model: {args.gpu_model})")
    print(f"CPU Endpoint: {args.cpu_url} (model: {args.cpu_model})")
    print("History Length: 20% more than forecast length (dynamic)")
    print(f"Forecast Lengths: {forecast_lengths}")
    print(f"Number of Scenarios: {num_scenarios_list}")
    print(
        f"Runs per test: {args.num_runs} (discarding first {args.num_warmup})"
    )
    print("=" * 60)

    results = []
    first_result = True  # Track if this is the first result for real-time table

    # Initialize CSV file if output is requested
    if args.output_csv:
        initialize_csv_file(args.output_csv)
        print(f"CSV output enabled: {args.output_csv}")

    # Run tests for each device
    for device in devices:
        device_name = device["name"]
        base_url = device["url"]
        model = device["model"]

        print(f"\n{'=' * 60}")
        print(f"Testing {device_name} endpoint: {base_url}")
        print(f"Model: {model}")
        print(f"{'=' * 60}")

        # Run tests for each forecast length
        for forecast_length in forecast_lengths:
            # Calculate history length as 20% more than forecast length
            history_length = int(forecast_length * 1.2)

            # Run tests for each number of scenarios
            for num_scenarios in num_scenarios_list:
                print(
                    f"\n  Testing: {device_name} | Forecast={forecast_length} | "
                    f"History={history_length} | Scenarios={num_scenarios}"
                )

                latencies, success = run_performance_test(
                    base_url=base_url,
                    model=model,
                    history_length=history_length,
                    forecast_length=forecast_length,
                    num_scenarios=num_scenarios,
                    num_warmup=args.num_warmup,
                    num_runs=args.num_runs,
                    api_key=args.api_key,
                )

                stats = calculate_statistics(latencies)

                result = {
                    "device": device_name,
                    "forecast_length": forecast_length,
                    "num_scenarios": num_scenarios,
                    "stats": stats,
                    "success": success,
                }
                results.append(result)

                # Save result to CSV immediately if requested
                if args.output_csv:
                    append_result_to_csv(result, args.output_csv)

                # Print real-time result
                if success and latencies:
                    valid_count = len(
                        [
                            latency
                            for latency in latencies
                            if latency != float("inf")
                        ]
                    )
                    print(
                        f"    ✅ Completed: {valid_count}/{len(latencies)} successful runs"
                    )
                    # Print formatted result in real-time
                    print_single_result(result, is_first=first_result)
                    if first_result:
                        first_result = False
                else:
                    print("    ❌ Failed: No successful runs")
                    # Still print the result row even if failed
                    print_single_result(result, is_first=first_result)
                    if first_result:
                        first_result = False

    # Print summary table
    print_results_table(results)

    # Save results to CSV if requested
    if args.output_csv:
        save_results_to_csv(results, args.output_csv)

    print("\n" + "=" * 60)
    print("Performance test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
