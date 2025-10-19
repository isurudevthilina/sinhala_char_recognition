import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_dir = os.path.join(ROOT, 'evaluation_results')
out_path = os.path.join(eval_dir, 'f1_score_distribution_boxplot.png')

print('Loading evaluation metrics...')

# Load F1 scores from CSV file
csv_path = os.path.join(eval_dir, 'detailed_classification_report.csv')
if not os.path.exists(csv_path):
    raise SystemExit(f'Detailed classification report not found at: {csv_path}')

df = pd.read_csv(csv_path)
print(f'Loaded CSV with columns: {list(df.columns)}')

# Extract F1 scores
if 'F1_Score' in df.columns:
    f1_scores = df['F1_Score'].values
elif 'f1-score' in df.columns:
    f1_scores = df['f1-score'].values
else:
    raise SystemExit(f'F1-score column not found. Available columns: {list(df.columns)}')

f1_scores = np.array(f1_scores)
print(f'Loaded F1 scores for {len(f1_scores)} classes')

# Calculate statistics
stats = {
    'min': np.min(f1_scores),
    'q1': np.percentile(f1_scores, 25),
    'median': np.median(f1_scores),
    'q3': np.percentile(f1_scores, 75),
    'max': np.max(f1_scores),
    'mean': np.mean(f1_scores),
    'std': np.std(f1_scores)
}

# IQR and outlier detection
iqr = stats['q3'] - stats['q1']
lower_fence = stats['q1'] - 1.5 * iqr
upper_fence = stats['q3'] + 1.5 * iqr
outliers_low = f1_scores[f1_scores < lower_fence]
outliers_high = f1_scores[f1_scores > upper_fence]

print(f'Statistics:')
print(f'  Min: {stats["min"]:.3f}')
print(f'  Q1: {stats["q1"]:.3f}')
print(f'  Median: {stats["median"]:.3f}')
print(f'  Q3: {stats["q3"]:.3f}')
print(f'  Max: {stats["max"]:.3f}')
print(f'  Mean: {stats["mean"]:.3f}')
print(f'  Std: {stats["std"]:.3f}')
print(f'  Outliers: {len(outliers_low)} low, {len(outliers_high)} high')

# Create the plot with updated seaborn style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
bp = ax1.boxplot(f1_scores, patch_artist=True, notch=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.6))

ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax1.set_title('F1-Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Add statistics text
stats_text = f'''Statistics:
Min: {stats["min"]:.3f}
Q1: {stats["q1"]:.3f}
Median: {stats["median"]:.3f}
Q3: {stats["q3"]:.3f}
Max: {stats["max"]:.3f}
Mean: {stats["mean"]:.3f}
Std: {stats["std"]:.3f}

Outliers: {len(outliers_low) + len(outliers_high)}
Classes: {len(f1_scores)}'''

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Histogram with performance categories
ax2.hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.3f}')
ax2.axvline(stats['median'], color='orange', linestyle='-', linewidth=2, label=f'Median: {stats["median"]:.3f}')

# Performance categories
excellent = np.sum(f1_scores >= 0.9)
good = np.sum((f1_scores >= 0.7) & (f1_scores < 0.9))
average = np.sum((f1_scores >= 0.5) & (f1_scores < 0.7))
poor = np.sum(f1_scores < 0.5)

# Add category lines
ax2.axvline(0.9, color='green', linestyle=':', alpha=0.7, label='Excellent (≥0.9)')
ax2.axvline(0.7, color='yellow', linestyle=':', alpha=0.7, label='Good (≥0.7)')
ax2.axvline(0.5, color='orange', linestyle=':', alpha=0.7, label='Average (≥0.5)')

ax2.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score Histogram with Performance Categories', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Add category counts
category_text = f'''Performance Distribution:
Excellent (≥0.9): {excellent} classes ({excellent/len(f1_scores)*100:.1f}%)
Good (0.7-0.9): {good} classes ({good/len(f1_scores)*100:.1f}%)
Average (0.5-0.7): {average} classes ({average/len(f1_scores)*100:.1f}%)
Poor (<0.5): {poor} classes ({poor/len(f1_scores)*100:.1f}%)'''

ax2.text(0.02, 0.98, category_text, transform=ax2.transAxes,
         verticalalignment='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'F1-score distribution plot saved to: {out_path}')

# Also create a violin plot version
fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6))

# Violin plot
parts = ax3.violinplot([f1_scores], positions=[1], showmeans=True, showmedians=True)
parts['bodies'][0].set_facecolor('lightcoral')
parts['bodies'][0].set_alpha(0.7)

# Overlay box plot
bp2 = ax3.boxplot([f1_scores], positions=[1], widths=0.1, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.8),
                  medianprops=dict(color='red', linewidth=2))

ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax3.set_title('F1-Score Distribution (Violin + Box Plot)', fontsize=14, fontweight='bold')
ax3.set_xlim(0.5, 1.5)
ax3.set_ylim(0, 1.05)
ax3.set_xticks([1])
ax3.set_xticklabels(['All Classes'])
ax3.grid(True, alpha=0.3)

# Add summary statistics
violin_stats = f'''Model Performance Summary:
Total Classes: {len(f1_scores)}
Mean F1-Score: {stats["mean"]:.3f}
Median F1-Score: {stats["median"]:.3f}
Standard Deviation: {stats["std"]:.3f}
Range: {stats["min"]:.3f} - {stats["max"]:.3f}

Performance Categories:
• Excellent (≥0.9): {excellent} ({excellent/len(f1_scores)*100:.1f}%)
• Good (0.7-0.9): {good} ({good/len(f1_scores)*100:.1f}%)
• Average (0.5-0.7): {average} ({average/len(f1_scores)*100:.1f}%)
• Poor (<0.5): {poor} ({poor/len(f1_scores)*100:.1f}%)'''

ax3.text(1.05, 0.5, violin_stats, transform=ax3.transData,
         verticalalignment='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
violin_path = os.path.join(eval_dir, 'f1_score_violin_plot.png')
plt.savefig(violin_path, dpi=200, bbox_inches='tight')
print(f'F1-score violin plot saved to: {violin_path}')

plt.close('all')
