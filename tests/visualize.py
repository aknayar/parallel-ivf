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


def clean_label(label):
    return label.replace("IVF", "").replace("Parallel", "Para").replace("Candidate", "C").replace("Query", "Q").replace("Scalar", "")


args = parse_args()
directory = args.directory
mode = args.mode
machine = directory.replace("../", "").split("/")[0].split("-")[0]

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
        "font.size": 24,
        "axes.labelsize": 28,
        "axes.titlesize": 30,
        "legend.fontsize": 22,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24   ,
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
    [
        "IVFCacheCandidateParallel",
        "IVFCacheQueryParallel",
        "IVFCacheSIMDCandidateParallel",
        "IVFCacheSIMDQueryParallel",
    ],
    [
        "IVFCacheV2QueryParallel",
        "IVFCacheV2SIMDQueryParallel",
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

all_indices = sorted(MODES[2])
palette = sns.color_palette("colorblind", n_colors=len(all_indices))

marker_cycle = itertools.cycle(
    ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "X", "d"]
)

style_map = {}
for name, color in zip(all_indices, palette):
    style_map[name] = {"color": color, "marker": next(marker_cycle)}

metrics = ["train_time", "build_time", "query_time"]
titles = ["Train", "Build", "Query"]
ylabels = ["Time (s)", "Time (s)", "Time (s)"]

# Combined plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

table_data = []

# Row 1: Time
for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    base_time, best_time = -1.0, float("inf")
    base_index, best_index = "None", "None"

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

        if index_name in ["IVFScalarCandidateParallel", "IVFScalarQueryParallel"]:
             val = df[df["n_threads"] == 1][metric].values
             if len(val) > 0 and val[0] > base_time:
                 base_time, base_index = val[0], index_name
        
        min_val = df[metric].min()
        if min_val < best_time:
            best_time, best_index = min_val, index_name

    ax.set_title(title)

    if base_time > 0:
        table_data.append({
            "metric": metric,
            "base_time": base_time,
            "base_index": clean_label(base_index),
            "best_time": best_time,
            "best_index": clean_label(best_index),
            "max speedup": base_time / best_time
        })

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

    if i == 1:
        ax.set_xlabel("# Threads")

    if i == 0:
        ax.set_ylabel("Speedup (×)")

    max_threads = max(max(data[index_name]["n_threads"]) for index_name in unique_indices)
    if max_threads == 128:
        ax.set_xticks([1, 16, 32, 64, 128])

    ax.set_ylim(bottom=0)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.5, label="Baseline")
    ax.grid(True, linestyle="--", alpha=0.6)

# Legend
handles, raw_labels = axes[0, 0].get_legend_handles_labels()
labels = [
    clean_label(label)
    for label in raw_labels
]
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5 if mode == 2 else 3 if mode == 1 else 2,
    frameon=False,
    fontsize=24,
)

plt.tight_layout(rect=[0, 0, 1, 0.93 if mode > 0 else .98], w_pad=0.25, h_pad=0.75)
output_path = os.path.normpath(os.path.join(directory, "../..", "plots", f"{machine}_{dataset}_combined_metrics_{mode}"))
plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
print(f"Saved plot to {output_path}")
plt.close()

if mode == 1:
    tables_dir = os.path.normpath(os.path.join(directory, "../..", "tables"))
    os.makedirs(tables_dir, exist_ok=True)
    output_csv_path = os.path.join(tables_dir, f"{machine}_{dataset}_speedup_table_{mode}.csv")

    df_table = pd.DataFrame(table_data)
    df_table.to_csv(output_csv_path, index=False)
    print(f"Saved speedup table to {output_csv_path}")
