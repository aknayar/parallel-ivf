import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import matplotlib.ticker
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize IVF performance results.")
    parser.add_argument(
        "directory", type=str, help="Path to the directory containing result .csv files"
    )
    parser.add_argument("mode", type=int, help="Plot mode")
    return parser.parse_args()


args = parse_args()
directory = args.directory
mode = args.mode

if not os.path.isdir(directory):
    print(f"Error: Directory '{directory}' does not exist.")
    exit(1)

csv_files = glob.glob(os.path.join(directory, "*.csv"))
if not csv_files:
    print(f"No .csv files found in '{directory}'.")
    exit(1)

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 22,
        "axes.labelsize": 26,
        "axes.titlesize": 30,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
)

DATASET_NAMES = {
    "gist": "GIST",
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "extreme": "Extreme",
}

MODES = [
    ["IVFSIMDQueryParallel", "IVFScalarQueryParallel"],
    [
        "IVFCacheV2QueryParallel",
        "IVFCacheV2SIMDQueryParallel",
        "IVFSIMDCandidateParallel",
        "IVFSIMDQueryParallel",
        "IVFScalarCandidateParallel",
        "IVFScalarQueryParallel",
    ],
    [
        "IVFCacheCandidateParallel",
        "IVFCacheQueryParallel",
        "IVFCacheSIMDCandidateParallel",
        "IVFCacheSIMDQueryParallel",
        "IVFCacheV2QueryParallel",
        "IVFCacheV2SIMDQueryParallel",
        "IVFSIMDCandidateParallel",
        "IVFSIMDQueryParallel",
        "IVFScalarCandidateParallel",
        "IVFScalarQueryParallel",
    ],
]

data = {}
dataset = None

for file_path in csv_files:
    filename = os.path.basename(file_path)

    # {dataset}_{index_name}.csv
    match = re.match(r"^([^_]+)_(.+)\.csv$", filename)
    if match:
        dataset, index_name = match.groups()

        if index_name not in MODES[mode]:
            continue

        df = pd.read_csv(file_path)
        data[index_name] = df

unique_indices = sorted(data.keys())
palette = sns.color_palette("colorblind", n_colors=len(unique_indices))

marker_cycle = itertools.cycle(
    ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "X", "d"]
)

style_map = {}
for name, color in zip(unique_indices, palette):
    style_map[name] = {"color": color, "marker": next(marker_cycle)}

metrics = ["train_time", "build_time", "query_time"]
titles = ["Train", "Build", "Query"]
ylabels = ["Time (s)", "Time (s)", "Time (s)"]

# Combined plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Row 1: Time
for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    ax = axes[0, i]
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.grid(which="major", linewidth=0.8)
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)

    for index_name in unique_indices:
        if index_name not in data:
            continue
        df = data[index_name].sort_values(by="n_threads")
        style = style_map[index_name]

        ax.plot(
            df["n_threads"],
            df[metric],
            marker=style["marker"],
            color=style["color"],
            label=index_name,
            linewidth=3.0,
            markersize=12,
        )

    ax.set_title(title)
    if i == 1:
        ax.set_xlabel("# Threads")

    if i == 0:
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.6)

# Row 2: Speedup
for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[1, i]

    ax.grid(which="major", linewidth=0.8)
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)

    for index_name in unique_indices:
        if index_name not in data:
            continue
        df = data[index_name].sort_values(by="n_threads")
        style = style_map[index_name]

        y_values = df[metric]
        baseline = y_values.iloc[0]
        y_values = baseline / y_values

        ax.plot(
            df["n_threads"],
            y_values,
            marker=style["marker"],
            color=style["color"],
            label=index_name,
            linewidth=3.0,
            markersize=12,
        )

    ax.set_title(title)
    if i == 1:
        ax.set_xlabel("# Threads")

    if i == 0:
        ax.set_ylabel("Speedup (×)")

    ax.set_ylim(bottom=0)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.5, label="Baseline")
    ax.grid(True, linestyle="--", alpha=0.6)

# Legend
handles, raw_labels = axes[0, 0].get_legend_handles_labels()
labels = [
    label.replace("IVF", "")
    .replace("Parallel", "Para")
    .replace("Candidate", "C")
    .replace("Query", "Q")
    for label in raw_labels
]
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5 if mode == 2 else 3 if mode == 1 else 2,
    frameon=False,
    fontsize=22,
)

plt.tight_layout(rect=[0, 0, 1, 0.95 if mode > 0 else .98], w_pad=0.25, h_pad=0.75)
output_path = os.path.join(directory, f"{dataset}_combined_metrics_{mode}")
plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"Saved plot to {output_path}")
plt.close()
