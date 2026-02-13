#!/usr/bin/env python3
"""
Performance test script for Synthefy Forecasting API using direct HTTP requests.

This script benchmarks forecasting performance by calling the OpenAPI endpoint
POST /api/v2/foundation_models/forecast/stream directly via the requests library
(no Synthefy client). Request/response shapes follow the OpenAPI 3.1 schema.
"""

import argparse
import csv
import os
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


FORECAST_STREAM_PATH = "/api/v2/foundation_models/forecast/stream"


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

    base_date = pd.Timestamp("2023-01-01")

    for scenario_idx in range(num_scenarios):
        history_dates = pd.date_range(
            base_date, periods=history_length, freq="D"
        )
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

        target_start = history_dates[-1] + pd.Timedelta(days=1)
        target_dates = pd.date_range(
            target_start, periods=forecast_length, freq="D"
        )
        target_data = {
            "date": target_dates,
            "sales": [np.nan] * forecast_length,
        }
        target_df = pd.DataFrame(target_data)
        target_dfs.append(target_df)

    return history_dfs, target_dfs


def _timestamp_to_str(ts: pd.Timestamp) -> str:
    """Format pandas Timestamp as YYYY-MM-DD string."""
    return ts.strftime("%Y-%m-%d")


def _values_to_json_list(values: Any) -> List[Optional[float]]:
    """Convert array-like to list with NaN -> None for JSON."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(float(v))
    return out


def build_forecast_v2_request(
    history_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    model: str,
    timestamp_col: str = "date",
    value_col: str = "sales",
) -> dict:
    """
    Build a ForecastV2Request body (OpenAPI schema) from history/target DataFrames.

    Parameters
    ----------
    history_dfs : List[pd.DataFrame]
        One DataFrame per scenario (history)
    target_dfs : List[pd.DataFrame]
        One DataFrame per scenario (target timestamps; values are NaN to forecast)
    model : str
        Model name
    timestamp_col : str
        Column name for timestamps
    value_col : str
        Column name for values

    Returns
    -------
    dict
        JSON-serializable body for POST /api/v2/foundation_models/forecast/stream
    """
    samples: List[List[dict]] = []

    for history_df, target_df in zip(history_dfs, target_dfs):
        history_ts = history_df[timestamp_col].apply(_timestamp_to_str).tolist()
        history_vals = _values_to_json_list(history_df[value_col])
        target_ts = target_df[timestamp_col].apply(_timestamp_to_str).tolist()
        target_vals = _values_to_json_list(target_df[value_col])

        payload = {
            "sample_id": value_col,
            "history_timestamps": history_ts,
            "history_values": history_vals,
            "target_timestamps": target_ts,
            "target_values": target_vals,
            "forecast": True,
            "metadata": False,
            "leak_target": False,
            "column_name": value_col,
        }
        # One scenario = one row of samples (one time series per row here)
        samples.append([payload])

    return {"samples": samples, "model": model}


def run_single_forecast_http(
    base_url: str,
    request_body: dict,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> float:
    """
    Perform a single POST to the forecast stream endpoint and return latency in seconds.

    Parameters
    ----------
    base_url : str
        Base URL (e.g. http://localhost:8014)
    request_body : dict
        ForecastV2Request body
    api_key : str or None
        Optional API key; sent as x-api-key header if set
    timeout : float
        Request timeout in seconds

    Returns
    -------
    float
        Elapsed time in seconds
    """
    url = base_url.rstrip("/") + FORECAST_STREAM_PATH
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    start = time.perf_counter()
    response = requests.post(
        url,
        json=request_body,
        headers=headers,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    return elapsed


def run_performance_test(
    base_url: str,
    model: str,
    history_length: int,
    forecast_length: int,
    num_scenarios: int,
    num_warmup: int = 5,
    num_runs: int = 15,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> Tuple[List[float], bool]:
    """
    Run performance test for a specific configuration using direct HTTP.

    Parameters
    ----------
    base_url : str
        Base URL of the API
    model : str
        Model name
    history_length, forecast_length, num_scenarios : int
        Data dimensions
    num_warmup : int
        Warmup runs to discard
    num_runs : int
        Total runs (latencies collected after warmup)
    api_key : str or None
        Optional API key
    timeout : float
        Request timeout in seconds

    Returns
    -------
    Tuple[List[float], bool]
        Latencies (excluding warmup) and success status
    """
    history_dfs, target_dfs = generate_test_data(
        history_length, forecast_length, num_scenarios
    )
    request_body = build_forecast_v2_request(
        history_dfs, target_dfs, model
    )

    latencies: List[float] = []
    success = False

    for i in range(num_runs):
        try:
            latency = run_single_forecast_http(
                base_url, request_body, api_key=api_key, timeout=timeout
            )
            if i >= num_warmup:
                latencies.append(latency)
            success = True
        except requests.RequestException as e:
            print(
                f"  ⚠️  Run {i + 1} failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            if i >= num_warmup:
                latencies.append(float("inf"))

    return latencies, success


def calculate_statistics(latencies: List[float]) -> dict:
    """Calculate mean, median, p95, std_dev, total from latency list."""
    if not latencies:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "std_dev": None,
            "total": None,
        }

    valid = [x for x in latencies if x != float("inf")]
    if not valid:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "std_dev": None,
            "total": None,
        }

    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "std_dev": float(np.std(valid)),
        "total": float(np.sum(valid)),
    }


def format_statistics(stats: dict) -> dict:
    """Format stats for display (numbers to strings)."""
    formatted = {}
    for key, value in stats.items():
        if value is None:
            formatted[key] = "N/A"
        elif value == float("inf"):
            formatted[key] = "INF"
        else:
            formatted[key] = f"{value:.4f}"
    return formatted


def print_single_result(result: dict, is_first: bool = False) -> None:
    """Print one result row; print header on first row."""
    header = (
        "Device | Forecast Length | Scenarios | Mean (s) | "
        "Median (s) | p95 (s) | Std Dev (s) | Total (s) | Req/min"
    )
    if is_first:
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

    device = result["device"]
    forecast_length = result["forecast_length"]
    num_scenarios = result["num_scenarios"]
    stats = format_statistics(result["stats"])
    rpm = throughput_requests_per_minute(result["stats"])
    rpm_str = f"{rpm:.1f}" if rpm is not None else "N/A"
    row = (
        f"{device:6s} | {forecast_length:15d} | {num_scenarios:9d} | "
        f"{stats['mean']:8s} | {stats['median']:9s} | {stats['p95']:6s} | "
        f"{stats['std_dev']:10s} | {stats['total']:8s} | {rpm_str:>7s}"
    )
    print(row)


def print_results_table(results: List[dict]) -> None:
    """Print summary table of all results."""
    header = (
        "Device | Forecast Length | Scenarios | Mean (s) | "
        "Median (s) | p95 (s) | Std Dev (s) | Total (s) | Req/min"
    )
    print("\n" + "=" * len(header))
    print("SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for result in results:
        device = result["device"]
        forecast_length = result["forecast_length"]
        num_scenarios = result["num_scenarios"]
        stats = format_statistics(result["stats"])
        rpm = throughput_requests_per_minute(result["stats"])
        rpm_str = f"{rpm:.1f}" if rpm is not None else "N/A"
        row = (
            f"{device:6s} | {forecast_length:15d} | {num_scenarios:9d} | "
            f"{stats['mean']:8s} | {stats['median']:9s} | {stats['p95']:6s} | "
            f"{stats['std_dev']:10s} | {stats['total']:8s} | {rpm_str:>7s}"
        )
        print(row)
    print("=" * len(header))


def throughput_requests_per_minute(stats: dict) -> Optional[float]:
    """Throughput as requests per minute (60 / mean latency). None if mean is missing or zero."""
    mean = stats.get("mean")
    if mean is None or mean <= 0 or mean == float("inf"):
        return None
    return 60.0 / mean


def get_csv_fieldnames() -> List[str]:
    return [
        "device",
        "forecast_length",
        "num_scenarios",
        "mean",
        "median",
        "p95",
        "std_dev",
        "total",
        "requests_per_minute",
        "success",
    ]


def initialize_csv_file(output_file: str) -> None:
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
            writer.writeheader()


def append_result_to_csv(result: dict, output_file: str) -> None:
    stats = result["stats"]
    rpm = throughput_requests_per_minute(stats)
    row = {
        "device": result["device"],
        "forecast_length": result["forecast_length"],
        "num_scenarios": result["num_scenarios"],
        "mean": stats.get("mean") if stats.get("mean") is not None else "",
        "median": stats.get("median") if stats.get("median") is not None else "",
        "p95": stats.get("p95") if stats.get("p95") is not None else "",
        "std_dev": stats.get("std_dev") if stats.get("std_dev") is not None else "",
        "total": stats.get("total") if stats.get("total") is not None else "",
        "requests_per_minute": f"{rpm:.4f}" if rpm is not None else "",
        "success": result["success"],
    }
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
        writer.writerow(row)
        f.flush()


def save_results_to_csv(results: List[dict], output_file: str) -> None:
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
        writer.writeheader()
        for result in results:
            stats = result["stats"]
            rpm = throughput_requests_per_minute(stats)
            row = {
                "device": result["device"],
                "forecast_length": result["forecast_length"],
                "num_scenarios": result["num_scenarios"],
                "mean": stats.get("mean") if stats.get("mean") is not None else "",
                "median": stats.get("median") if stats.get("median") is not None else "",
                "p95": stats.get("p95") if stats.get("p95") is not None else "",
                "std_dev": stats.get("std_dev") if stats.get("std_dev") is not None else "",
                "total": stats.get("total") if stats.get("total") is not None else "",
                "requests_per_minute": f"{rpm:.4f}" if rpm is not None else "",
                "success": result["success"],
            }
            writer.writerow(row)
    print(f"\nResults saved to CSV: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance test for Synthefy Forecasting API (direct HTTP)"
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
        help="Model for GPU endpoint (default: sfm-moe-v1)",
    )
    parser.add_argument(
        "--cpu-model",
        type=str,
        default="sfm-moe-v1",
        help="Model for CPU endpoint (default: sfm-moe-v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (sent as x-api-key header)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Warmup runs to discard (default: 5)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=15,
        help="Runs per test (default: 15)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to CSV file for results",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    devices = [
        {"name": "GPU", "url": args.gpu_url, "model": args.gpu_model},
        {"name": "CPU", "url": args.cpu_url, "model": args.cpu_model},
    ]
    forecast_lengths = [2, 10, 100]
    num_scenarios_list = [1, 2, 4]

    print("Synthefy Forecasting API Performance Test (HTTP)")
    print("=" * 60)
    print(f"Endpoint: POST ...{FORECAST_STREAM_PATH}")
    print(f"GPU: {args.gpu_url} (model: {args.gpu_model})")
    print(f"CPU: {args.cpu_url} (model: {args.cpu_model})")
    print("History Length: 20% more than forecast length")
    print(f"Forecast Lengths: {forecast_lengths}")
    print(f"Scenarios: {num_scenarios_list}")
    print(f"Runs per test: {args.num_runs} (warmup: {args.num_warmup})")
    print("=" * 60)

    results: List[dict] = []
    first_result = True

    if args.output_csv:
        initialize_csv_file(args.output_csv)
        print(f"CSV output: {args.output_csv}")

    for device in devices:
        device_name = device["name"]
        base_url = device["url"]
        model = device["model"]

        print(f"\n{'=' * 60}")
        print(f"Testing {device_name} endpoint: {base_url}")
        print(f"Model: {model}")
        print(f"{'=' * 60}")

        for forecast_length in forecast_lengths:
            history_length = int(forecast_length * 1.2)
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
                    timeout=args.timeout,
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

                if args.output_csv:
                    append_result_to_csv(result, args.output_csv)

                if success and latencies:
                    valid_count = sum(1 for x in latencies if x != float("inf"))
                    print(
                        f"    ✅ Completed: {valid_count}/{len(latencies)} successful runs"
                    )
                else:
                    print("    ❌ Failed: No successful runs")

                print_single_result(result, is_first=first_result)
                if first_result:
                    first_result = False

    print_results_table(results)
    if args.output_csv:
        save_results_to_csv(results, args.output_csv)

    print("\n" + "=" * 60)
    print("Performance test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
