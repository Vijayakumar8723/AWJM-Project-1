import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============ DataSet ============
surface_data = {
    'Exp.value': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46,
                  4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'Ensemble': [4.65,3.23,4.43,5.01,3.87,5.05,4.34,4.65,3.97,3.57,4.04,4.52,5.56,3.65,4.16,4.71,4.43,4.34,3.43,4.25,
                 4.56,5.35,3.35,4.43,4.58,5.24,3.85]
}

mrr_data = {
    'Exp.value': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95,
                  148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66,
                  139.85],
    'Ensemble': [118.15,128.31,120.37,114.03,128.86,98.40,119.45,112.40,133.38,122.63,143.81,122.22,108.80,150.03,103.46,
                 110.29,120.37,117.62,146.87,125.88,126.50,100.84,140.92,120.37,106.65,105.56,141.33]

}

# ============ DataFrames ============
datasets = {
    "Surface Roughness": pd.DataFrame(surface_data),
    "Material Removal Rate": pd.DataFrame(mrr_data),
}
# ============ Font Settings ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# ============ Color scheme ============

colors = ['#1f77b4', '#ff7f0e']  # Blue: Actual, Orange: Predicted

# ============ Plot Density & Summary ============
def plot_and_summarize(df, metric_name):
    plt.figure(figsize=(6,4))
    sns.kdeplot(df['Exp.value'], label='Actual', color=colors[0], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(df['Ensemble'], label='Predicted', color=colors[1], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)

# ============ Title and Axis Label ============
    plt.title(f"{metric_name}: Actual vs Predicted Density",fontsize=16, fontweight='bold', fontname='DejaVu Sans')
    plt.xlabel(metric_name, fontsize=14, fontweight='bold', fontname='DejaVu Sans')
    plt.ylabel('Density', fontsize=14, fontweight='bold', fontname='DejaVu Sans')
    plt.xticks(fontsize=12, fontweight='bold', fontname='DejaVu Sans')
    plt.yticks(fontsize=12, fontweight='bold', fontname='DejaVu Sans')
    plt.legend(fontsize=10, title_fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============ Statistical Summary ============
    print(f"\n=== {metric_name} Statistical Summary ===")
    print(f"Actual   - Mean: {df['Exp.value'].mean():.4f}, Std: {df['Exp.value'].std():.4f}")
    print(f"Predicted- Mean: {df['Ensemble'].mean():.4f}, Std: {df['Ensemble'].std():.4f}")

    ks_stat, p_value = stats.ks_2samp(df['Exp.value'], df['Ensemble'])
    print(f"Kolmogorov-Smirnov Test: Stat={ks_stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("Distributions are statistically similar (p > 0.05)")
    else:
        print("Distributions are statistically different (p ≤ 0.05)")

# ============ Apply Function to All Datasets ============
for name, df in datasets.items():
    plot_and_summarize(df, name)
