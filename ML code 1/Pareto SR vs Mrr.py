import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches

# ============ Dataset ============
data = {
    'pressure MPa': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300,300, 220, 140, 300, 140, 180, 220, 220, 300, 220,220, 140, 300, 220, 140, 140, 300],
    'standoff distance mm': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1,3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2, 2.5, 1, 1, 2, 1, 3, 3],
    'traverse speed mm min': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64,96, 88, 96, 96, 64, 80, 80, 80, 64, 80, 80, 96, 96, 80, 96, 64, 64],
    'mass flow rate kg min': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface roughness um': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'material removal rate mm3 min': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'kerf angle deg': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
}
df = pd.DataFrame(data)
ra = df['surface roughness um'].values
mrr = df['material removal rate mm3 min'].values

# ============ Pareto-Optimal Identification ============

def identify_pareto_optimal(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(scores[is_efficient] < c, axis=1) |
                np.any(scores[is_efficient] > c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([ra, -mrr])
pareto_mask = identify_pareto_optimal(scores)
pareto_points = np.column_stack([ra[pareto_mask], mrr[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

best_quality_idx = np.argmin(ra)
best_productivity_idx = np.argmax(mrr)
balanced_idx = np.argmin(np.sqrt((ra - ra.min())**2 + (mrr - mrr.max())**2))

# ============ Plot ===========
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.scatter(ra, mrr, color='gray', alpha=0.6, s=50, label='Experimental Points')
plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black', linewidth=1,
            label='Pareto-optimal Solutions')
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.5)
key_pts = [
    (best_quality_idx, "Best Quality", 's', 'red'),
    (best_productivity_idx, "Best Productivity", '^', 'green'),
    (balanced_idx, "Balanced", 'D', 'purple')
]
for idx, label, marker, color in key_pts:
    x, y = ra[idx], mrr[idx]
    plt.scatter(x, y, s=150, marker=marker, color=color, edgecolors='black', linewidth=2)
    plt.annotate(f"{label}\nRa={x:.2f}, MRR={y:.2f}",
                 xy=(x, y), xytext=(20, -40),
                 textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.85),
                 arrowprops=dict(arrowstyle='->', lw=1.2))
if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices, 0], pareto_points[hull.vertices, 1],
             'blue', alpha=0.1, label='Feasible Region')

# ============ Title and Axis Label ============
plt.xlabel('Surface Roughness Ra (μm)', fontsize=14, fontweight='bold')
plt.ylabel('Material Removal Rate (mm³/min)', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for AWJM Process\nTrade-off between Ra and MRR',fontsize=16, fontweight='bold', pad=20)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# =============== Summary Tables ==========================
print(f"Total Pareto-optimal solutions: {pareto_mask.sum()}")
print(f"Best Quality → Ra={ra[best_quality_idx]:.2f}, MRR={mrr[best_quality_idx]:.2f}")
print(f"Best Productivity → Ra={ra[best_productivity_idx]:.2f}, MRR={mrr[best_productivity_idx]:.2f}")
print(f"Balanced → Ra={ra[balanced_idx]:.2f}, MRR={mrr[balanced_idx]:.2f}")

best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Pressure (MPa)',
        'Standoff Distance (mm)',
        'Traverse Speed (mm/min)',
        'Mass Flow Rate (kg/min)',
        'Surface Roughness (μm)',
        'Material Removal Rate (mm³/min)',
        'Kerf Angle (deg)'        # <-- missing parameter added
    ],
    'Best Quality (Min Ra)': best_quality.values,
    'Best Productivity (Max MRR)': best_productivity.values,
    'Balanced': balanced.values
})


print("\n=========== Table 3 ===========")
print(summary_table.to_string(index=False))
print("==============================================")
