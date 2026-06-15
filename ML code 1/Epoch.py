import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ============ Actual Values ============
surface_actual = np.array([
    4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.50,
    3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
    4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.80
])
mrr_actual = np.array([
    119.98, 126.36, 122.40, 111.57, 126.63, 97.63, 121.60, 109.81,
    137.08, 117.28, 143.30, 125.70, 106.95, 148.41, 102.02, 115.20,
    122.40, 119.69, 145.92, 131.74, 132.71, 99.70, 140.98, 122.40,
    104.26, 103.66, 139.85
])

# ============ Predicted Values ============
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

# ============ Ensemble ============
datasets = {
    "Surface Roughness": (surface_actual + surface_pred) / 2,
    "Material Removal Rate": (mrr_actual + mrr_pred) / 2,
}
# ============ sequence index ============
num_samples = len(surface_actual)
X = np.arange(num_samples).reshape(-1, 1)

# ============ Train NN and plot training loss============
def train_and_plot_loss(y_ensemble, target_name):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    history = model.fit(X, y_ensemble, epochs=500, batch_size=4, verbose=0, callbacks=[es])
    loss = np.array(history.history['loss'])
    smooth_loss = np.convolve(loss, np.ones(10)/10, mode='valid')

    # ============ Title and Axis Label ============
    plt.figure(figsize=(6, 3.8))
    plt.plot(smooth_loss, linewidth=1.5)
    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Square Error", fontsize=14, fontweight="bold")
    plt.xticks(fontweight='bold',fontsize=12,fontname='Dejavu Sans')
    plt.yticks(fontweight='bold', fontsize=12, fontname='Dejavu Sans')
    plt.title(f"{target_name} - Training Loss", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ============ run for all datasets ============
for name, ensemble_values in datasets.items():
    train_and_plot_loss(ensemble_values, name)
