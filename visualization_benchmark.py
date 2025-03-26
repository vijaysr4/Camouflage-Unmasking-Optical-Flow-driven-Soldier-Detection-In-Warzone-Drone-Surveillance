import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Adjust as needed:
    PER_FRAME_CSV = 'output/benchmark_all/per_frame_metrics.csv'
    OVERALL_CSV = 'output/benchmark_all/overall_metrics.csv'
    PLOT_OUT = 'output/benchmark_all_plots'
    os.makedirs(PLOT_OUT, exist_ok=True)

    # Load CSV data
    df_frames = pd.read_csv(PER_FRAME_CSV)
    df_over = pd.read_csv(OVERALL_CSV)

    print("Loaded per-frame CSV, shape:", df_frames.shape)
    print("Loaded overall CSV, shape:", df_over.shape)

    # Overall histogram of EPE (all frames)
    epe_all_vals = df_frames['epe_all'].dropna()
    hist_path = os.path.join(PLOT_OUT, 'epe_all_distribution.png')

    plt.figure(figsize=(10, 6))
    plt.hist(epe_all_vals, bins=100, edgecolor='black', alpha=0.75)
    plt.axvline(epe_all_vals.mean(), linestyle='--', label=f"Mean = {epe_all_vals.mean():.2f}")
    plt.axvline(epe_all_vals.median(), linestyle=':', label=f"Median = {epe_all_vals.median():.2f}")
    plt.title("EPE_all Distribution (All Scenes & Frames)", fontsize=16)
    plt.xlabel("EPE (pixels)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()
    print(f"Saved EPE-all histogram to {hist_path}")


    # Bar chart of overall means
    metrics = df_over['metric'].values
    means = df_over['overall_mean'].values
    bar_over_path = os.path.join(PLOT_OUT, 'overall_means.png')

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, means, edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("Overall Mean Metrics (All Scenes & Frames)", fontsize=16)
    plt.ylabel("Mean EPE (pixels)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height + 0.02, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(bar_over_path, bbox_inches='tight')
    plt.close()
    print(f"Saved overall mean metrics bar chart to {bar_over_path}")

    # Scene-level average of epe_all
    grouped_scene = df_frames.groupby('scene')['epe_all'].mean().dropna().sort_values()
    bar_scene_path = os.path.join(PLOT_OUT, 'scene_epe_all_bar.png')

    plt.figure(figsize=(12, 6))
    bars = grouped_scene.plot(kind='bar', edgecolor='black', linewidth=0.8)
    plt.title("Average EPE_all per Scene", fontsize=16)
    plt.ylabel("Mean EPE (pixels)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    for idx, val in enumerate(grouped_scene):
        plt.text(idx, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(bar_scene_path, bbox_inches='tight')
    plt.close()
    print(f"Saved epe_all per scene bar chart to {bar_scene_path}")

    print("All plots saved in:", PLOT_OUT)

if __name__ == '__main__':
    main()
