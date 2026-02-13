#!/usr/bin/env python3
"""
Generate a customer-ready chart from results_gpu8000_cpu8001.csv.
Plots throughput (requests per minute) for GPU vs CPU by forecast length and scenario count.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["requests_per_minute"] = pd.to_numeric(df["requests_per_minute"], errors="coerce")
    df = df.dropna(subset=["requests_per_minute"])
    return df


def create_chart(df: pd.DataFrame, output_path: str, title: str | None = None) -> None:
    forecast_lengths = sorted(df["forecast_length"].unique())
    scenario_counts = sorted(df["num_scenarios"].unique())
    x = np.arange(len(forecast_lengths))
    n_scenarios = len(scenario_counts)
    total_width = 0.8
    scenario_group_width = total_width / n_scenarios
    bar_width = scenario_group_width / 2

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = {"GPU": "#2563eb", "CPU": "#dc2626"}  # blue, red

    for i, scenario in enumerate(scenario_counts):
        scenario_base = (
            x - (total_width / 2) + (i * scenario_group_width) + (scenario_group_width / 2)
        )
        for j, device in enumerate(["GPU", "CPU"]):
            device_offset = (j - 0.5) * bar_width
            values = []
            for fl in forecast_lengths:
                subset = df[
                    (df["forecast_length"] == fl)
                    & (df["num_scenarios"] == scenario)
                    & (df["device"] == device)
                ]
                values.append(subset.iloc[0]["requests_per_minute"] if not subset.empty else 0)
            label = device if i == 0 else ""
            bars = ax.bar(
                scenario_base + device_offset,
                values,
                bar_width,
                label=label,
                color=colors[device],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.8,
            )
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{int(round(value))}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

    ax.set_xlabel("Forecast horizon", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput (requests per minute)", fontsize=14, fontweight="bold")
    ax.set_title(
        title or "GPU vs CPU pool — Throughput by forecast length and scenario count",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fl}" for fl in forecast_lengths], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(title="Pool", title_fontsize=12, fontsize=11, loc="upper right", framealpha=0.95)
    ax.set_ylim(bottom=0)
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max * 1.12)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.text(
        0.99,
        0.02,
        "Higher is better  •  Model: migas-1.5",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        style="italic",
        color="gray",
    )
    # Scenario labels below x-axis
    trans = ax.get_xaxis_transform()
    for i, scenario in enumerate(scenario_counts):
        scenario_base = (
            x - (total_width / 2) + (i * scenario_group_width) + (scenario_group_width / 2)
        )
        for fl_idx in range(len(forecast_lengths)):
            ax.text(
                scenario_base[fl_idx],
                -0.06,
                f"{scenario} scenario{'s' if scenario != 1 else ''}",
                ha="center",
                va="top",
                fontsize=10,
                transform=trans,
                color="gray",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chart from GPU/CPU results CSV")
    parser.add_argument(
        "csv",
        nargs="?",
        default="results_gpu8000_cpu8001.csv",
        help="Input CSV (default: results_gpu8000_cpu8001.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default="performance_chart_gpu8000_cpu8001.png",
        help="Output image path",
    )
    parser.add_argument("--title", default=None, help="Chart title")
    args = parser.parse_args()
    df = load_csv(args.csv)
    create_chart(df, args.output, args.title)


if __name__ == "__main__":
    main()
