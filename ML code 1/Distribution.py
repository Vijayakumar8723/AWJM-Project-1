import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============ DataSet ============
data = {
    'pressure_MPa': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300,
                     300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                     220, 140, 300, 220, 140, 140, 300],
    'standoff_distance_mm': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1,
                             3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2,
                             2.5, 1, 1, 2, 1, 3, 3],
    'traverse_speed_mm_min': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64,
                              96, 88, 96, 96, 64, 80, 80, 80, 64, 80,
                              80, 96, 96, 80, 96, 64, 64],
    'mass_flow_rate_kg_min': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,
                              0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,
                              0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                             3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                             4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'material_removal_rate': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,
                                      143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,
                                      132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],

}

# ============ Dataframe ============
df = pd.DataFrame(data)

# ============ target variables ============
targets = ['surface_roughness', 'material_removal_rate']

# ============ Train/Test split ============
train_test_data = {}
for target in targets:
    train, test = train_test_split(df[target], test_size=0.2, random_state=42)
    train_test_data[target] = (train, test)

# ============ font style ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# ============ Plot histograms ============
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

colors = {
    'surface_roughness': ('pink', 'brown'),
    'material_removal_rate': ('skyblue', 'royalblue'),

}

for i, target in enumerate(targets):
    train, test = train_test_data[target]
    axes[i].hist(train, bins=8, alpha=0.7, color=colors[target][0],
                 edgecolor='black', label='Train')
    axes[i].hist(test, bins=8, alpha=0.7, color=colors[target][1],
                 edgecolor='black', label='Test')

    # ============ Title and Axis Label ============

    axes[i].set_title(f'{target.replace("_", " ").title()} Distribution',
                      fontsize=16, fontweight='bold', fontfamily='DejaVu Sans')
    axes[i].set_xlabel(target.replace("_", " ").title(),
                       fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    axes[i].set_ylabel('Frequency',
                       fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('DejaVu Sans')
        label.set_fontsize(12)
    axes[i].legend(fontsize=12)
plt.tight_layout()
plt.show()
