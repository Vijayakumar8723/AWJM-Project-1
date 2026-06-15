import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ============ FUNCTION TO PLOT LEARNING CURVES ============
def plot_learning_curve(X, y, title):
    ridge = Ridge()
    rf = RandomForestRegressor(random_state=0)
    xgb = XGBRegressor(random_state=0, verbosity=0)
    nn = MLPRegressor(random_state=0, max_iter=1000)

    ensemble = VotingRegressor([
        ('ridge', ridge),
        ('rf', rf),
        ('xgb', xgb),
        ('nn', nn)
    ])
    train_sizes, train_scores, test_scores = learning_curve(
        ensemble, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5)
    )
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation Score')

    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color='g')

    # ============ Title and Axis Label ============
    plt.xlabel("Training Examples", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Squared Error", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="best", fontsize=12)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()
# ============ SURFACE ROUGHNESS DATA ============
surface_data = {
    'Exp_value':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.50,3.97,4.63,5.56,
                 3.62,4.08,4.75,4.46,4.41,3.42,4.27,4.67,5.36,3.29,4.46,4.53,5.23,3.80],
    'Ridge':[4.49,3.37,4.34,5.03,3.98,4.96,4.26,4.70,3.85,3.65,4.33,4.42,5.64,3.72,
             4.35,4.67,4.34,4.25,3.38,4.19,4.43,5.30,3.37,4.34,4.69,5.31,3.99],
    'RandomForest':[4.83,3.39,4.46,4.90,3.78,5.06,4.38,4.61,3.91,3.62,3.93,4.57,5.45,
                    3.56,4.24,4.68,4.46,4.40,3.45,4.35,4.60,5.30,3.39,4.46,4.60,5.20,3.79],
    'XGBoost':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.50,3.97,4.63,5.56,
               3.62,4.08,4.75,4.46,4.41,3.42,4.27,4.67,5.36,3.29,4.46,4.53,5.23,3.80],
    'NeuralNet':[4.47,3.17,4.40,5.07,3.93,5.11,4.30,4.68,3.92,3.56,4.05,4.40,5.63,
                 3.71,4.08,4.70,4.40,4.27,3.43,4.19,4.48,5.43,3.35,4.40,4.57,5.26,3.89]
}
X1 = np.column_stack([surface_data['Ridge'], surface_data['RandomForest'],
                      surface_data['XGBoost'], surface_data['NeuralNet']])
y1 = np.array(surface_data['Exp_value'])
plot_learning_curve(X1, y1, "Learning Curve - Ensemble Surface Roughness")

# ============ MRR DATA ============
mrr = {
    'Exp_value':[119.98,126.36,122.40,111.57,126.63,97.63,121.60,109.81,137.08,117.28,
                  143.30,125.70,106.95,148.41,102.02,115.20,122.40,119.69,145.92,131.74,
                  132.71,99.70,140.98,122.40,104.26,103.66,139.85],
    'Ridge Regression':[120.20,126.70,122.19,117.72,130.56,94.96,121.22,113.82,134.08,
                        126.66,141.46,123.17,109.76,149.42,102.93,114.27,122.19,119.47,
                        145.52,124.18,124.92,98.87,138.53,122.19,106.83,105.86,137.55],
    'RandomForest':[120.80,130.02,122.64,110.71,127.61,98.92,121.74,109.21,132.05,122.34,
                    141.74,124.28,106.82,146.05,102.65,107.63,122.64,121.00,144.99,129.38,
                    128.66,100.41,137.34,122.64,104.17,105.10,140.65],
    'XGBoost':[119.98,126.36,122.40,111.57,126.63,97.63,121.60,109.81,137.08,117.28,
               143.30,125.70,106.95,148.41,102.02,115.20,122.40,119.69,145.92,131.74,
               132.71,99.70,140.98,122.40,104.26,103.66,139.85],
    'NeuralNet':[113.10,129.65,115.53,117.42,131.27,100.46,114.50,116.93,130.43,126.20,
                  147.24,116.57,111.83,155.28,105.86,105.60,115.53,111.80,150.06,117.96,
                  119.27,103.35,145.04,115.53,111.00,107.69,145.27]
}
df_mrr = pd.DataFrame(mrr)
X2 = df_mrr[['Ridge Regression', 'RandomForest', 'XGBoost', 'NeuralNet']].values
y2 = df_mrr['Exp_value'].values

plot_learning_curve(X2, y2, "Learning Curve - Ensemble MRR")

