#!/usr/bin/env python3
"""
Parse src_result_merged.tex latency table and generate per-dataset line charts.
Each chart: x = n (data scale), y = latency (s), one line per baseline.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ============================================================
# 1. Hardcoded data extracted from src_result_merged.tex
#    (n values for DL/Image: [1000, 2000, 5000, 10000], Reasoning: [1, 5, 10, 20])
# ============================================================

n_dl = [1000, 2000, 5000, 10000]
n_reasoning = [1, 5, 10, 20]

# Color / style palette for baselines
STYLE = {
    'evadb':       {'color': '#d73027', 'marker': 's', 'ls': '--'},
    'morphingdb':  {'color': '#fc8d59', 'marker': 'o', 'ls': '-'},
    'gendb':       {'color': '#fee08b', 'marker': '^', 'ls': '-.'},
    'hermes':      {'color': '#91bfdb', 'marker': 'D', 'ls': '--'},
    'neurdb':      {'color': '#4575b4', 'marker': 'v', 'ls': '-.'},
    'db_agent':    {'color': '#1a1a1a', 'marker': '*', 'ls': '-', 'lw': 2.5},
}

DATASETS = {
    # --- Deep Learning (Series) ---
    'slice': {
        'category': 'Deep Learning (Series)',
        'n': n_dl,
        'data': {
            'evadb':      [5.42,   6.29,   9.03,   11.89],
            'morphingdb': [0.639,  0.736,  0.95,   1.304],
            'gendb':      [0.221,  0.484,  0.663,  0.956],
            'hermes':     [0.125,  0.137,  0.257,  0.464],
            'neurdb':     [0.077,  0.140,  0.307,  0.592],
            'db_agent':   [0.037,  0.076,  0.179,  0.328],
        },
    },
    # --- Deep Learning (NLP) ---
    'sst-2': {
        'category': 'Deep Learning (NLP)',
        'n': n_dl,
        'data': {
            'evadb':      [85.42,  177.61,  431.97,  864.62],
            'morphingdb': [4.39,   5.95,    10.78,   18.85],
            'gendb':      [36.37,  70.19,   170.15,  346.13],
            'hermes':     [4.479,  6.576,   10.90,   18.95],
            'neurdb':     [4.444,  8.896,   22.32,   44.68],
            'db_agent':   [4.29,   6.000,   10.64,   18.6],
        },
    },
    # --- Image ---
    'imagenet': {
        'category': 'Image',
        'n': n_dl,
        'data': {
            'evadb':      [77.2,   155.7,   377.84,  737.35],
            'morphingdb': [11.86,  13.06,   18.21,   28.1],
            'gendb':      [12.45,  22.39,   54.71,   106.07],
            'hermes':     [5.237,  10.15,   24.36,   47.08],
            'neurdb':     [12.22,  23.57,   58.70,   116.00],
            'db_agent':   [4.71,   6.38,    12.69,   23.13],
        },
    },
    'cifar-10': {
        'category': 'Image',
        'n': n_dl,
        'data': {
            'evadb':      [31.58,  62.75,   154.57,  301.56],
            'morphingdb': [12.03,  14.22,   21.22,   33.3],
            'gendb':      [11.19,  23.02,   58.85,   113.26],
            'hermes':     [4.519,  7.148,   14.32,   27.16],
            'neurdb':     [10.44,  21.52,   53.18,   104.60],
            'db_agent':   [4.49,   6.74,    13.31,   24.71],
        },
    },
    # --- Reasoning ---
    'HotpotQA': {
        'category': 'Reasoning',
        'n': n_reasoning,
        'data': {
            'morphingdb': [5.8696, 20.9274, 38.6813, 76.1707],
            'hermes':     [5.4713, 18.9648, 36.1208, 69.8286],
            'neurdb':     [5.5876, 19.8220, 35.9892, 71.0315],
            'gendb':      [5.6607, 20.3130, 39.0735, 75.3437],
            'db_agent':   [5.0388, 18.3130, 33.2352, 65.4884],
        },
    },
    'musique': {
        'category': 'Reasoning',
        'n': n_reasoning,
        'data': {
            'morphingdb': [4.30,  18.60,  37.42,  72.79],
            'hermes':     [3.82,  17.72,  31.27,  66.54],
            'neurdb':     [3.82,  17.81,  31.99,  66.00],
            'gendb':      [3.79,  19.16,  40.30,  79.87],
            'db_agent':   [3.87,  16.65,  30.40,  64.43],
        },
    },
}

# Display name mapping
DISPLAY_NAME = {
    'evadb': 'EvaDB',
    'morphingdb': 'MorphingDB',
    'gendb': 'GenDB',
    'hermes': 'Hermes',
    'neurdb': 'NeurDB',
    'db_agent': 'DBAgent',
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_single_dataset(name, info, save_path):
    """Plot one line chart for a single dataset."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    n_vals = info['n']
    x_ticks = [str(v) for v in n_vals]

    for method, values in info['data'].items():
        s = STYLE.get(method, {'color': 'gray', 'marker': 'o', 'ls': '-'})
        ax.plot(
            n_vals, values,
            label=DISPLAY_NAME.get(method, method),
            color=s['color'],
            marker=s['marker'],
            linestyle=s['ls'],
            linewidth=s.get('lw', 1.5),
            markersize=6,
        )

    ax.set_xlabel('Number of tuples ($n$)', fontsize=10)
    ax.set_ylabel('Latency (s)', fontsize=10)
    ax.set_xticks(n_vals)
    ax.set_xticklabels(x_ticks)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_title(f'{name} ({info["category"]})', fontsize=11, fontweight='bold')

    # Use log scale if range is large
    max_val = max(v for vals in info['data'].values() for v in vals)
    min_val = min(v for vals in info['data'].values() for v in vals)
    if max_val / max(min_val, 1e-9) > 50:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_all_grid(save_path):
    """Plot all datasets in a 2x3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    datasets_list = list(DATASETS.items())

    for idx, (name, info) in enumerate(datasets_list):
        ax = axes[idx]
        n_vals = info['n']
        x_ticks = [str(v) for v in n_vals]

        for method, values in info['data'].items():
            s = STYLE.get(method, {'color': 'gray', 'marker': 'o', 'ls': '-'})
            ax.plot(
                n_vals, values,
                label=DISPLAY_NAME.get(method, method),
                color=s['color'],
                marker=s['marker'],
                linestyle=s['ls'],
                linewidth=s.get('lw', 1.5),
                markersize=5,
            )

        ax.set_xlabel('Number of tuples ($n$)', fontsize=10)
        ax.set_ylabel('Latency (s)', fontsize=10)
        ax.set_xticks(n_vals)
        ax.set_xticklabels(x_ticks)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')

        # Auto log scale check
        max_val = max(v for vals in info['data'].values() for v in vals)
        min_val = min(v for vals in info['data'].values() for v in vals)
        if max_val / max(min_val, 1e-9) > 50:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Shared legend in the bottom-right unused subplot area
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Individual dataset plots ===")
    for name, info in DATASETS.items():
        fname = f"latency_{name.lower().replace('-', '_')}.png"
        plot_single_dataset(name, info, os.path.join(OUTPUT_DIR, fname))

    print("\n=== Combined 2x3 grid ===")
    plot_all_grid(os.path.join(OUTPUT_DIR, "latency_all_datasets.png"))

    print("\nDone.")


if __name__ == '__main__':
    main()
