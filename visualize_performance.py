#!/usr/bin/env python3
"""
Visualization script for Synthefy Forecasting API performance test results.

This script parses terminal output from performance tests and generates
a grouped bar plot comparing GPU vs CPU performance across forecast lengths
and scenario counts.
"""

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_terminal_output(text: str) -> pd.DataFrame:
    """
    Parse terminal output to extract performance data.

    Parameters
    ----------
    text : str
        Terminal output text containing performance test results

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: device, forecast_length, num_scenarios, median_latency
    """
    # Pattern to match table rows: Device | Forecast Length | Scenarios | Mean | Median | ...
    pattern = (
        r"(GPU|CPU)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)"
    )

    data = []
    for line in text.split("\n"):
        match = re.search(pattern, line)
        if match:
            device = match.group(1)
            forecast_length = int(match.group(2))
            num_scenarios = int(match.group(3))
            # mean = float(match.group(4))  # Not used for now
            median_latency = float(match.group(5))

            data.append(
                {
                    "device": device,
                    "forecast_length": forecast_length,
                    "num_scenarios": num_scenarios,
                    "median_latency": median_latency,
                }
            )

    return pd.DataFrame(data)


def load_csv_results(csv_file: str) -> pd.DataFrame:
    """
    Load performance test results from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to CSV file containing performance test results

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: device, forecast_length, num_scenarios, median_latency
    """
    df = pd.read_csv(csv_file)

    # Convert median column to median_latency for compatibility
    if "median" in df.columns:
        df["median_latency"] = pd.to_numeric(df["median"], errors="coerce")
    elif "median_latency" not in df.columns:
        raise ValueError(
            "CSV file must contain either 'median' or 'median_latency' column"
        )

    # Ensure required columns exist
    required_columns = [
        "device",
        "forecast_length",
        "num_scenarios",
        "median_latency",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file missing required column: {col}")

    # Filter out rows with missing median_latency (failed tests)
    df = df.dropna(subset=["median_latency"])

    return df.loc[:, required_columns]


def create_visualization(
    df: pd.DataFrame, output_file: str = "performance_comparison.png"
):
    """
    Create a grouped bar plot comparing GPU vs CPU performance.

    X-axis: Forecast lengths (10, 100, 1000)
    Y-axis: Median latency (seconds)
    Hue: Device type (GPU vs CPU)
    Multiple bars per forecast length for different scenario counts (1, 2, 4, 8, 16)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    output_file : str
        Output filename for the plot
    """
    # Set style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))

    # Group by forecast length on x-axis
    # Within each forecast length, show grouped bars for each scenario count
    # Each scenario bar has GPU and CPU side by side (hue)

    forecast_lengths = sorted(df["forecast_length"].unique())
    scenario_counts = sorted(df["num_scenarios"].unique())

    # Calculate bar positions
    x = np.arange(len(forecast_lengths))

    # Width calculations
    # Each forecast length group has multiple scenario bars
    # Each scenario bar has 2 bars (GPU and CPU)
    n_scenarios = len(scenario_counts)
    total_width = 0.8  # Total width for all bars in one forecast length group
    scenario_group_width = (
        total_width / n_scenarios
    )  # Width for one scenario group
    bar_width = scenario_group_width / 2  # Width for one GPU/CPU bar

    # Colors
    colors = {"GPU": "#2E86AB", "CPU": "#A23B72"}

    # Plot bars for each scenario count
    for i, scenario in enumerate(scenario_counts):
        # Base position for this scenario group within the forecast length
        scenario_base = (
            x
            - (total_width / 2)
            + (i * scenario_group_width)
            + (scenario_group_width / 2)
        )

        # Plot GPU and CPU bars for this scenario
        for j, device in enumerate(["GPU", "CPU"]):
            # Position offset for GPU (left) or CPU (right) within the scenario group
            device_offset = (j - 0.5) * bar_width

            # Get values for this device and scenario across all forecast lengths
            values = []
            for fl in forecast_lengths:
                subset = df[
                    (df["forecast_length"] == fl)
                    & (df["num_scenarios"] == scenario)
                    & (df["device"] == device)
                ]
                if not subset.empty:
                    values.append(subset.iloc[0]["median_latency"])
                else:
                    values.append(0)

            # Label only for first scenario to avoid duplicates in legend
            label = device if i == 0 else ""

            bars = ax.bar(
                scenario_base + device_offset,
                values,
                bar_width,
                label=label,
                color=colors[device],
                alpha=0.85 if device == "GPU" else 0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            
            # Add latency labels on top of each bar
            for bar, value in zip(bars, values):
                if value > 0:  # Only label non-zero bars
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

    # Customize the plot
    ax.set_xlabel("Forecast Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("Median Latency (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Performance Comparison: GPU vs CPU\nMedian Latency by Forecast Length and Scenario Count",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(forecast_lengths)

    # Add "Lower is better" note in top right corner
    ax.text(
        0.98,
        0.98,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Add legend
    ax.legend(
        title="Device",
        title_fontsize=11,
        fontsize=10,
        loc="upper left",
        framealpha=0.9,
    )

    # Add grid for easier reading
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Set y-axis to start from 0 for better comparison
    # Add top margin to accommodate bar labels
    ax.set_ylim(bottom=0)
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max * 1.1)  # Add 10% margin at top for labels

    # Add scenario count annotations below x-axis labels
    # This helps identify which bars correspond to which scenario count
    scenario_labels = [f"Scenarios: {', '.join(map(str, scenario_counts))}"]
    ax.text(
        0.5,
        -0.08,
        "  ".join(scenario_labels),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_file}")

    # Show the plot (only if display is available)
    try:
        plt.show()
    except Exception:
        # If display is not available (e.g., headless environment), just close
        plt.close()


def main():
    """Main function to parse terminal output and create visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize performance test results from terminal output"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to file containing terminal output or CSV file (default: read from stdin or use embedded data)",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV file containing performance test results (alternative to --input)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_comparison.png",
        help="Output filename for the plot (default: performance_comparison.png)",
    )

    args = parser.parse_args()

    # Load data from CSV if specified
    if args.input_csv:
        df = load_csv_results(args.input_csv)
    elif args.input and args.input.endswith(".csv"):
        df = load_csv_results(args.input)
    else:
        # Read terminal output
        if args.input:
            with open(args.input, "r") as f:
                text = f.read()
        else:
            # Try reading from stdin if it has data, otherwise use embedded data
            if not sys.stdin.isatty():
                stdin_text = sys.stdin.read().strip()
                text = stdin_text if stdin_text else None
            else:
                text = None

            # If no text from stdin, use embedded terminal output data from the user's selection
            # Data extracted from terminal output lines 139-216
            if not text:
                text = """GPU    |              10 |         1 | 3.0258   | 2.9808    | 3.2900 | 0.1543     | 30.2582 
GPU    |              10 |         2 | 5.4820   | 5.4289    | 5.7542 | 0.1564     | 54.8196 
GPU    |              10 |         4 | 12.2634  | 12.2096   | 12.8199 | 0.3821     | 122.6342
GPU    |              10 |         8 | 23.0935  | 23.1069   | 23.8998 | 0.5996     | 230.9353
GPU    |              10 |        16 | 44.4370  | 42.2846   | 53.7876 | 4.7165     | 444.3696
CPU    |              10 |         1 | 12.7294  | 12.5555   | 13.8985 | 0.7538     | 127.2940
CPU    |              10 |         2 | 30.2321  | 30.5623   | 31.3706 | 0.9006     | 302.3209
CPU    |              10 |         4 | 49.3681  | 48.7006   | 53.5604 | 2.2408     | 493.6808
CPU    |              10 |         8 | 197.8664 | 222.5304  | 245.5288 | 47.5904    | 1978.6637"""

        # Parse the data
        df = parse_terminal_output(text)

    if df.empty:
        print("Error: No data found in terminal output", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(df)} data points")
    print(f"Forecast lengths: {sorted(df['forecast_length'].unique())}")
    print(f"Scenario counts: {sorted(df['num_scenarios'].unique())}")
    print(f"Devices: {sorted(df['device'].unique())}")

    # Create visualization
    create_visualization(df, args.output)


if __name__ == "__main__":
    main()
