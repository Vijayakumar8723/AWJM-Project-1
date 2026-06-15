import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============ Dataset ============
surface = pd.DataFrame({
    'Actual': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46,
               4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'Predicted': [4.65, 3.23, 4.43, 5.01, 3.87, 5.05, 4.34, 4.65, 3.97, 3.57, 4.04, 4.52, 5.56, 3.65, 4.16, 4.71, 4.43,
                 4.34, 3.43, 4.25, 4.56, 5.35, 3.35, 4.43, 4.58, 5.24, 3.85]
})

mrr = pd.DataFrame({
    'Actual': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95,
               148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66,
               139.85],
    'Predicted': [118.15, 128.31, 120.37, 114.03, 128.86, 98.40, 119.45, 112.40, 133.38, 122.63, 143.81, 122.22, 108.80,
                 150.03, 103.46, 110.29, 120.37, 117.62, 146.87, 125.88, 126.50, 100.84, 140.92, 120.37, 106.65, 105.56,
                 141.33]

})

# ============ Output Parameter ============
datasets = {
    "Surface Roughness ": surface,
    "Material Removal Rate": mrr,
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# ============ Heatmap Plot ============
for name, df in datasets.items():
    plt.figure(figsize=(12, 4))
    heat_matrix = np.vstack([df['Actual'].values, df['Predicted'].values])
    im = plt.imshow(
        heat_matrix,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )

# ============ Title and Axis Label ============
    plt.title(f'{name}: Actual vs Predicted Heat Map', fontsize=16,fontweight='bold',fontfamily='DejaVu Sans')
    plt.xlabel('Sample Index',fontsize=14,fontweight='bold',fontfamily='DejaVu Sans')
    plt.yticks([0, 1], ['Actual', 'Predicted'],fontsize=14,fontweight='bold',fontfamily='DejaVu Sans' )
    plt.xticks(fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
    plt.yticks(fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')

# ============ Colorbar with Label ============
    cbar = plt.colorbar(im)
    cbar.set_label(
        'Value',
        fontsize=18,
        fontweight='bold',
        fontfamily='DejaVu Sans'
    )
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('DejaVu Sans')

# ============ Style & Layout ============
    plt.grid(False)
    plt.tight_layout()
    plt.show()
