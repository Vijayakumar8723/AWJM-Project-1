import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============ Input Parameters ============
inputs = {
    'AP': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300,
                     300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                     220, 140, 300, 220, 140, 140, 300],
    'ND ': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1,
                             3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2,
                             2.5, 1, 1, 2, 1, 3, 3],
    'CS': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64,
                              96, 88, 96, 96, 64, 80, 80, 80, 64, 80,
                              80, 96, 96, 80, 96, 64, 64],
    'AMFR': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,
                              0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,
                              0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35]
}

# ============ Output Parameters ============

outputs = {
    'SR': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                          3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                          4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'MRR': [119.98,126.36,122.4,111.57,126.63,97.63,121.6,109.81,137.08,117.28,
                              143.3,125.7,106.95,148.41,102.02,115.2,122.4,119.69,145.92,131.74,
                              132.71,99.7,140.98,122.4,104.26,103.66,139.85],

}
# ============ DataFrame ============
df = pd.DataFrame({**inputs, **outputs})

# ============ Compute correlation matrix ============
corr_matrix = df.corr()

# ============ font style ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ============ Plot heatmap ============
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
# ============ Title and Axis Label ============
plt.title("Correlation Matrix", fontsize=16, fontweight='bold', fontfamily='DejaVu Sans', pad=15)
plt.xticks(rotation=90, ha='right', fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
plt.yticks(rotation=0, fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
plt.tight_layout()
plt.show()
