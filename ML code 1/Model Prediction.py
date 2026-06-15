import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============ Dataset ============
data = {
    'pressure': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300, 300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                 220, 140, 300, 220, 140, 140, 300],
    'standoff': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1, 3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2, 2.5, 1, 1, 2, 1, 3, 3],
    'traverse': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64, 96, 88, 96, 96, 64, 80, 80, 80, 64, 80, 80, 96, 96, 80, 96, 64, 64],
    'mass_flow': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35, 0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45,
                  0.45, 0.55, 0.5, 0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08, 4.75,
                          4.46, 4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'mrr': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95, 148.41,
            102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],

}

df = pd.DataFrame(data)
exp_nos = list(range(1, len(df) + 1))

# ============ Input & Target ============
X = df[['pressure', 'standoff', 'traverse', 'mass_flow']]
targets = {
    'Surface Roughness': df['surface_roughness'],
    'Material Removal Rate': df['mrr'],

}

# ============ Models ============
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=120, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=120, random_state=42)
}

# ============ Neural Network ============
nn_models = {
    'Surface Roughness': MLPRegressor(hidden_layer_sizes=(6,4,3), max_iter=2000, random_state=42),
    'Material Removal Rate': MLPRegressor(hidden_layer_sizes=(8,5,4), max_iter=2000, random_state=42),

}

colors = ['red', 'blue', 'green', 'orange']

# ============ Function To Generate Full Predictions ============
def get_predictions(X, y, target_name):
    predictions = {}

    full_models = models.copy()
    full_models['Neural Network'] = nn_models[target_name]

    for name, model in full_models.items():

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        if name == 'Neural Network':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        full_pred = np.zeros(len(X))
        full_pred[X_test.index] = y_pred
        full_pred[X_train.index] = y.iloc[X_train.index].values

        predictions[name] = full_pred

    return predictions

# ============ FIGURES ============
for target_name, y in targets.items():
    predictions = get_predictions(X, y, target_name)
    plt.figure(figsize=(12, 6))

    # Actual experimental values
    plt.plot(exp_nos, y, 'ko-', linewidth=3, markersize=7,
             label='Experimental', markerfacecolor='black')

    # Model predictions
    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(exp_nos, pred, 's-', color=colors[i], markersize=5,
                 label=name, alpha=0.85, linewidth=2)

    plt.title(f"{target_name} - Model Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Experiment Number", fontsize=14, fontweight='bold')
    plt.ylabel(target_name, fontsize=14, fontweight='bold')
    plt.xticks(fontweight='bold', fontsize=12, fontname='Dejavu Sans')
    plt.yticks(fontweight='bold', fontsize=12, fontname='Dejavu Sans')
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=10)
    plt.show()
