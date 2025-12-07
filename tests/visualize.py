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

data = {}
difficulty = None

for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    # {difficulty}_{index_name}.csv
    match = re.match(r"^([^_]+)_(.+)\.csv$", filename)
    difficulty, index_name = match.groups()
        
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

for metric, title, ylabel in zip(metrics, titles, ylabels):
    plt.figure(figsize=(14, 10))
    plt.yscale('log')
    # plt.ylim(.1 if metric == 'train_time' else 1 if metric == 'query_time' else .9, 100)
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.grid(which="major", linewidth=0.8)
    plt.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    
    for index_name in unique_indices:
        if index_name not in data:
            continue
            
        df = data[index_name]

        df_sorted = df.sort_values(by='n_threads')
        
        style = style_map[index_name]
        label_name = index_name
        
        plt.plot(df_sorted['n_threads'], df_sorted[metric], 
                    marker=style['marker'], 
                    color=style['color'],
                    label=label_name,
                    linewidth=3.0,
                    markersize=12)

    plt.title(f"{title} vs Threads ({difficulty})")
    plt.xlabel("Number of Threads")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(directory, f"{metric}")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved plot to {output_path}")
    plt.close()
