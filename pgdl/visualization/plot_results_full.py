#!/usr/bin/env python3
"""
Parse full latency table (all 11 datasets: 3 Series + 3 NLP + 3 Image + 2 Reasoning)
and generate a 4x3 grid of line charts.
Each subplot: x = n (data scale), y = avg latency per entry (s), one line per baseline.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ============================================================
# 1. Data extracted from main_result.tex (9 base) + reasoning_result.tex (2)
# ============================================================

N_DL = [1000, 2000, 5000, 10000]
N_REASONING = [1, 5, 10, 20]

# Color / style palette for baselines
STYLE = {
    'evadb':       {'color': '#d73027', 'marker': 's', 'ls': '--'},
    'morphingdb':  {'color': '#fc8d59', 'marker': 'o', 'ls': '-'},
    'gendb':       {'color': '#fdae61', 'marker': '^', 'ls': '-.'},
    'hermes':      {'color': '#1a9850', 'marker': 'D', 'ls': '--'},
    'neurdb':      {'color': '#4575b4', 'marker': 'v', 'ls': '-.'},
    'db_agent':    {'color': '#1a1a1a', 'marker': '*', 'ls': '-', 'lw': 2.5},
}

DISPLAY_NAME = {
    'evadb': 'EvaDB',
    'morphingdb': 'MorphingDB',
    'gendb': 'GenDB',
    'hermes': 'Hermes',
    'neurdb': 'NeurDB',
    'db_agent': 'DBAgent',
}

DATASETS = {
    # --- SERIES (n=1000,2000,5000,10000) ---
    'slice': {
        'category': 'Series',
        'n': N_DL,
        'data': {
            'evadb':      [5.42,   6.29,   9.03,   11.89],
            'morphingdb': [0.639,  0.736,  0.95,   1.304],
            'gendb':      [0.221,  0.484,  0.663,  0.956],
            'hermes':     [0.125,  0.137,  0.257,  0.464],
            'neurdb':     [0.077,  0.140,  0.307,  0.592],
            'db_agent':   [0.037,  0.076,  0.179,  0.328],
        },
    },
    'swarm': {
        'category': 'Series',
        'n': N_DL,
        'data': {
            'morphingdb': [0.657,  0.782,  1.18,   1.686],
            'gendb':      [0.499,  0.540,  0.622,  0.811],
            'hermes':     [0.340,  0.645,  1.495,  3.023],
            'neurdb':     [0.452,  0.838,  2.008,  4.082],
            'db_agent':   [0.071,  0.153,  0.389,  0.864],
        },
    },
    'year_predict': {
        'category': 'Series',
        'n': N_DL,
        'data': {
            'evadb':      [1.85,   1.95,   2.27,   3.26],
            'morphingdb': [0.591,  0.61,   0.796,  1.077],
            'gendb':      [0.108,  0.113,  0.118,  0.144],
            'hermes':     [0.029,  0.055,  0.170,  0.262],
            'neurdb':     [0.025,  0.046,  0.117,  0.234],
            'db_agent':   [0.022,  0.040,  0.137,  0.204],
        },
    },
    # --- NLP (n=1000,2000,5000,10000) ---
    'sst-2': {
        'category': 'NLP',
        'n': N_DL,
        'data': {
            'evadb':      [85.42,  177.61,  431.97,  864.62],
            'morphingdb': [4.39,   5.95,    10.78,   18.85],
            'gendb':      [36.37,  70.19,   170.15,  346.13],
            'hermes':     [4.479,  6.576,   10.90,   18.95],
            'neurdb':     [4.444,  8.896,   22.32,   44.68],
            'db_agent':   [4.29,   6.000,   10.64,   18.6],
        },
    },
    'imdb': {
        'category': 'NLP',
        'n': N_DL,
        'data': {
            'evadb':      [91.5,   171.65,  416.35,  853.87],
            'morphingdb': [4.44,   5.95,    10.77,   18.84],
            'gendb':      [36.51,  69.96,   175.92,  359.56],
            'hermes':     [4.218,  5.451,   10.68,   19.22],
            'neurdb':     [5.846,  8.790,   22.01,   44.25],
            'db_agent':   [4.29,   6.000,   10.66,   18.97],
        },
    },
    'financial': {
        'category': 'NLP',
        'n': N_DL,
        'data': {
            'evadb':      [25.72,  47.7,    120.17,  224.68],
            'morphingdb': [6.05,   6.89,    9.59,    14.0],
            'gendb':      [25.76,  49.29,   52.36,   100.43],
            'hermes':     [3.585,  7.340,   12.58,   22.60],
            'neurdb':     [4.885,  8.386,   11.05,   22.08],
            'db_agent':   [5.98,   6.79,    9.53,    13.73],
        },
    },
    # --- IMAGE (n=1000,2000,5000,10000) ---
    'imagenet': {
        'category': 'Image',
        'n': N_DL,
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
        'n': N_DL,
        'data': {
            'evadb':      [31.58,  62.75,   154.57,  301.56],
            'morphingdb': [12.03,  14.22,   21.22,   33.3],
            'gendb':      [11.19,  23.02,   58.85,   113.26],
            'hermes':     [4.519,  7.148,   14.32,   27.16],
            'neurdb':     [10.44,  21.52,   53.18,   104.60],
            'db_agent':   [4.49,   6.74,    13.31,   24.71],
        },
    },
    'stanford_dogs': {
        'category': 'Image',
        'n': N_DL,
        'data': {
            'evadb':      [12.81,  25.39,   60.8,    119.46],
            'morphingdb': [14.94,  17.41,   24.32,   35.24],
            'gendb':      [5.87,   10.80,   25.05,   48.91],
            'hermes':     [6.62,   10.25,   14.95,   29.21],
            'neurdb':     [5.020,  10.16,   25.36,   49.73],
            'db_agent':   [5.91,   8.11,    14.95,   26.35],
        },
    },
    # --- REASONING (n=1,5,10,20) ---
    'HotpotQA': {
        'category': 'Reasoning',
        'n': N_REASONING,
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
        'n': N_REASONING,
        'data': {
            'morphingdb': [4.30,  18.60,  37.42,  72.79],
            'hermes':     [3.82,  17.72,  31.27,  66.54],
            'neurdb':     [3.82,  17.81,  31.99,  66.00],
            'gendb':      [3.79,  19.16,  40.30,  79.87],
            'db_agent':   [3.87,  16.65,  30.40,  64.43],
        },
    },
}

# Ordering for 4x3 grid: Series(3) | NLP(3) | Image(3) | Reasoning(2) + blank
GRID_ORDER = [
    'slice', 'swarm', 'year_predict',
    'sst-2', 'imdb', 'financial',
    'imagenet', 'cifar-10', 'stanford_dogs',
    'HotpotQA', 'musique',
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_full_grid(save_path):
    """Plot all 11 datasets in a 4x3 grid."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 11.5))
    axes = axes.flatten()

    for idx, name in enumerate(GRID_ORDER):
        info = DATASETS[name]
        ax = axes[idx]
        n_vals = info['n']
        x_ticks = [str(v) for v in n_vals]

        for method, values in info['data'].items():
            s = STYLE.get(method, {'color': 'gray', 'marker': 'o', 'ls': '-'})
            # Convert total latency to avg latency per entry
            avg_values = [v / n for v, n in zip(values, n_vals)]
            ax.plot(
                n_vals, avg_values,
                label=DISPLAY_NAME.get(method, method),
                color=s['color'],
                marker=s['marker'],
                linestyle=s['ls'],
                linewidth=s.get('lw', 1.5),
                markersize=5,
            )

        ax.set_xlabel('Number of tuples ($n$)', fontsize=10)
        ax.set_ylabel('Avg Latency per Entry (s)', fontsize=10)
        ax.set_xticks(n_vals)
        ax.set_xticklabels(x_ticks)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')

        # Category label in top-left of each subplot
        ax.text(0.02, 0.95, info['category'], transform=ax.transAxes,
                fontsize=8, fontweight='bold', color='gray',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7))

        # Use log scale for all subplots
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # --- Throughput grouped bar chart in the 12th subplot ---
    ax_tp = axes[-1]
    task_types = ['Series', 'NLP', 'Image']
    # Throughput data: {method: [series_tp, nlp_tp, image_tp]}
    # EvaDB excluded (no throughput data in any category)
    TP_DATA = {
        'morphingdb': [12800, 256, 608],
        'gendb':      [12800, 288, 800],
        'hermes':     [12800, 288, 512],
        'neurdb':     [12800, 320, 1056],
        'db_agent':   [12800, 320, 1056],
    }
    tp_methods = ['morphingdb', 'gendb', 'hermes', 'neurdb', 'db_agent']
    n_methods = len(tp_methods)
    bar_width = 0.13
    x_pos = [0, 1, 2]

    for i, method in enumerate(tp_methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        s = STYLE.get(method, {'color': 'gray'})
        vals = TP_DATA[method]
        bars = ax_tp.bar(
            [p + offset for p in x_pos], vals,
            bar_width, label=DISPLAY_NAME.get(method, method),
            color=s['color'], edgecolor='black', linewidth=0.5,
        )

    # Add value labels: skip Series (all same), label NLP and Image individually
    for gi, group in enumerate(task_types):
        for i, method in enumerate(tp_methods):
            offset = (i - n_methods / 2 + 0.5) * bar_width
            val = TP_DATA[method][gi]
            cx = x_pos[gi] + offset
            label = '12800+' if val >= 10000 else str(val)
            if gi == 0:
                # Series: all same value, label once centered
                if i == n_methods // 2:
                    ax_tp.text(x_pos[gi], 13000, label,
                               ha='center', va='center', fontsize=8, fontweight='bold')
            else:
                ax_tp.text(cx, val, label,
                           ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax_tp.set_xticks(x_pos)
    ax_tp.set_xticklabels(task_types, fontsize=10, fontweight='bold')
    ax_tp.set_yscale('log')
    ax_tp.set_ylim(100, 20000)
    ax_tp.set_ylabel('Max Throughput (concurrent tasks)', fontsize=10)
    ax_tp.set_title('Throughput Comparison', fontsize=11, fontweight='bold')
    ax_tp.grid(True, alpha=0.3, linestyle='--', axis='y', which='both')
    ax_tp.legend(fontsize=6, loc='upper right', ncol=1, framealpha=0.9)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.005), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_full_grid(os.path.join(OUTPUT_DIR, 'latency_per_entry_all_11_datasets.png'))
    print("Done.")


if __name__ == '__main__':
    main()
