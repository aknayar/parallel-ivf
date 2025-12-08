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
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize IVF performance results.")
    parser.add_argument("directory", type=str, help="Path to the directory containing result .csv files")
    return parser.parse_args()

args = parse_args()
directory = args.directory

if not os.path.isdir(directory):
    print(f"Error: Directory '{directory}' does not exist.")
    exit(1)

csv_files = glob.glob(os.path.join(directory, "*.csv"))
if not csv_files:
    print(f"No .csv files found in '{directory}'.")
    exit(1)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm", 
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 30,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22
})

DATASET_NAMES = {
    "gist": "GIST",
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "extreme": "Extreme",
}

data = {}
dataset = None

for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    # {dataset}_{index_name}.csv
    match = re.match(r"^([^_]+)_(.+)\.csv$", filename)
    if match:
        dataset, index_name = match.groups()
        
        df = pd.read_csv(file_path)
        data[index_name] = df

unique_indices = sorted(data.keys())
palette = sns.color_palette("colorblind", n_colors=len(unique_indices))

marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'd'])

style_map = {}
for name, color in zip(unique_indices, palette):
    style_map[name] = {'color': color, 'marker': next(marker_cycle)}

metrics = ['train_time', 'build_time', 'query_time']
titles = ['Train Time', 'Build Time', 'Query Time']
ylabels = ['Time (s)', 'Time (s)', 'Time (s)']

speedup_metrics = ['train_speedup', 'build_speedup', 'query_speedup']
speedup_titles = ['Train Speedup', 'Build Speedup', 'Query Speedup']

for metric, title, ylabel, speedup_metric, speedup_title in zip(metrics, titles, ylabels, speedup_metrics, speedup_titles):
    
    # Time Plot
    plt.figure(figsize=(7, 5))
    plt.yscale('log')
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.grid(which="major", linewidth=0.8)
    plt.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    
    for index_name in unique_indices:
        if index_name not in data: continue
        df = data[index_name].sort_values(by='n_threads')
        style = style_map[index_name]
        
        plt.plot(df['n_threads'], df[metric], 
                    marker=style['marker'], color=style['color'], label=index_name,
                    linewidth=3.0, markersize=12)

    plt.title(f"{title} vs. # Threads ({DATASET_NAMES.get(dataset, dataset)})")
    plt.xlabel("# Threads")
    plt.ylabel(ylabel)
    # plt.legend()  # Removed legend from individual plot
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(directory, f"{dataset}_{metric}")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved plot to {output_path}")
    plt.close()

    # Speedup Plot
    plt.figure(figsize=(14, 10))
    plt.grid(which="major", linewidth=0.8)
    plt.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    
    for index_name in unique_indices:
        if index_name not in data: continue
        df = data[index_name].sort_values(by='n_threads')
        
        baseline = df[metric].iloc[0]
        speedup = baseline / df[metric]
        
        style = style_map[index_name]
        
        plt.plot(df['n_threads'], speedup, 
                    marker=style['marker'], color=style['color'], label=index_name,
                    linewidth=3.0, markersize=12)

    plt.title(f"{speedup_title} vs. # Threads ({DATASET_NAMES.get(dataset, dataset)})")
    plt.xlabel("# Threads")
    plt.ylabel("Speedup (×)")
    # plt.legend()  # Removed legend from individual plot
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(directory, f"{dataset}_{speedup_metric}")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved plot to {output_path}")
    plt.close()


# Tiled plots
configs = [
    {
        "filename": "time_metrics_tiled",
        "metrics_list": metrics,
        "titles_list": titles,
        "ylabels_list": ylabels,
        "is_speedup": False
    },
    {
        "filename": "speedup_metrics_tiled",
        "metrics_list": metrics,
        "titles_list": speedup_titles,
        "ylabels_list": ["Speedup"] * 3,
        "is_speedup": True
    }
]

for config in configs:
    fig, axes = plt.subplots(1, 3, figsize=(42, 10))
    
    for ax, metric, title, ylabel in zip(axes, config["metrics_list"], config["titles_list"], config["ylabels_list"]):
        if not config["is_speedup"]:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
        ax.grid(which="major", linewidth=0.8)
        ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
        
        for index_name in unique_indices:
            if index_name not in data: continue
            df = data[index_name].sort_values(by='n_threads')
            style = style_map[index_name]
            
            y_values = df[metric]
            if config["is_speedup"]:
                baseline = y_values.iloc[0]
                y_values = baseline / y_values
            
            ax.plot(df['n_threads'], y_values, 
                    marker=style['marker'], color=style['color'], label=index_name,
                    linewidth=3.0, markersize=12)
        
        ax.set_title(f"{title} vs. # Threads ({DATASET_NAMES.get(dataset, dataset)})")
        ax.set_xlabel("# Threads")
        ax.set_ylabel(ylabel)
        # ax.legend() # Removed legend from tiled plot
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(directory, f"{dataset}_{config["filename"]}")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved plot to {output_path}")
    plt.close()

# Combined plot
fig, axes = plt.subplots(2, 3, figsize=(24, 15))

for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    ax = axes[0, i]
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    ax.grid(which="major", linewidth=0.8)
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    
    for index_name in unique_indices:
        if index_name not in data: continue
        df = data[index_name].sort_values(by='n_threads')
        style = style_map[index_name]
        
        ax.plot(df['n_threads'], df[metric], 
                marker=style['marker'], color=style['color'], label=index_name,
                linewidth=3.0, markersize=12)

    ax.set_title(f"{title} vs. # Threads ({DATASET_NAMES.get(dataset, dataset)})")
    ax.set_xlabel("# Threads")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)

for i, (metric, title) in enumerate(zip(metrics, speedup_titles)):
    ax = axes[1, i]
    
    ax.grid(which="major", linewidth=0.8)
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    
    for index_name in unique_indices:
        if index_name not in data: continue
        df = data[index_name].sort_values(by='n_threads')
        style = style_map[index_name]
        
        y_values = df[metric]
        baseline = y_values.iloc[0]
        y_values = baseline / y_values
        
        ax.plot(df['n_threads'], y_values, 
                marker=style['marker'], color=style['color'], label=index_name,
                linewidth=3.0, markersize=12)

    ax.set_title(f"{title} vs. # Threads ({DATASET_NAMES.get(dataset, dataset)})")
    ax.set_xlabel("# Threads")
    ax.set_ylabel("Speedup (×)")
    ax.grid(True, linestyle='--', alpha=0.6)

# Legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=len(unique_indices) // 2, frameon=False, fontsize=22)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Make space for legend
output_path = os.path.join(directory, f"{dataset}_combined_metrics_2x3")
plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved plot to {output_path}")
plt.close()

# Generate legend
legend_handles = []
legend_labels = []
for index_name in unique_indices:
    if index_name not in data: continue
    style = style_map[index_name]
    line = Line2D([0], [0], color=style['color'], marker=style['marker'], 
                  linewidth=3.0, markersize=12, label=index_name)
    legend_handles.append(line)
    legend_labels.append(index_name)

fig_legend = plt.figure(figsize=(len(unique_indices) * 3, 2))
legend = fig_legend.legend(handles=legend_handles, labels=legend_labels, loc='center', ncol=len(unique_indices) // 2, frameon=False)

output_path = os.path.join(directory, f"{dataset}_legend")
fig_legend.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
fig_legend.savefig(output_path + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
print(f"Saved legend to {output_path}")
plt.close(fig_legend)
