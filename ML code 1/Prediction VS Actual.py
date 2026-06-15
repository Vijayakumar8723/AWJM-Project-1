import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============ Actual values ============

surface_actual = np.array([
    4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.50,
    3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
    4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.80
])

mrr_actual = np.array([
    119.98, 126.36, 122.40, 111.57, 126.63, 97.63, 121.60, 109.81, 137.08,
    117.28, 143.30, 125.70, 106.95, 148.41, 102.02, 115.20, 122.40,
    119.69, 145.92, 131.74, 132.71, 99.70, 140.98, 122.40, 104.26,
    103.66, 139.85
])


# ============ Predicted values ============

surface_pred = np.array([
    4.65, 3.25, 4.43, 5.02, 3.89, 5.03, 4.34, 4.65, 3.90, 3.58,
    4.05, 4.52, 5.56, 3.64, 4.17, 4.71, 4.43, 4.34, 3.45, 4.25,
    4.57, 5.36, 3.34, 4.43, 4.58, 5.25, 3.83
])

mrr_pred = np.array([
    117.92, 127.71, 120.20, 114.28, 129.46, 98.73, 119.33, 111.19,
    133.08, 122.76, 143.58, 122.12, 109.42, 150.70, 104.27, 110.69,
    120.20, 117.54, 147.84, 125.59, 126.23, 101.32, 141.08, 120.20,
    105.96, 106.23, 139.99
])

# ============ Performance metrics ============

targets = {
    "Surface Roughness": (surface_actual, surface_pred),
    "Material Removal Rate": (mrr_actual, mrr_pred),
}

performance_metrics = {}
for name, (y_true, y_pred) in targets.items():
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    performance_metrics[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# ============ Plot function ============

def plot_actual_vs_pred(y_true, y_pred, title, metrics):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(y_true, y_pred, color="blue", edgecolor="k", alpha=0.7, s=60)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal Fit")

    # ============ Title and Axis Label ============

    ax.set_title(f"{title} – Actual vs Predicted",fontsize=16, fontweight="bold", fontname="DejaVu Sans")
    ax.set_xlabel("Actual Values", fontsize=14, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylabel("Predicted Values", fontsize=14, fontweight="bold", fontname="DejaVu Sans")
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):label.set_fontweight("bold")
    metrics_text = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nR²: {metrics['R²']:.4f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_actual_vs_pred(surface_actual, surface_pred, "Surface Roughness",
                    performance_metrics["Surface Roughness"])

plot_actual_vs_pred(mrr_actual, mrr_pred, "Material Removal Rate",
                    performance_metrics["Material Removal Rate"])
#============ Summary Print ============
print("\nEnsemble Model Performance Summary")
print("=" * 55)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 55)
for name, m in performance_metrics.items():
    print(f"{name:<25} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")
print("=" * 55)
