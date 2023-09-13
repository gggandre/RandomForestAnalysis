#----------------------------------------------------------
# Feedback Moment: Module 2
# Improve the implementation of a machine learning technique with the use of a framework.
#
# Date: 11-Sep-2023
# Author:
#           A01753176 Gilberto André García Gaytán
#
# Ensure that this libraries are installed:
# pip install scikit-learn
# pip install pandas
# pip install matplotlib
# pip install numpy
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv('top5_leagues_player.csv')
data = data.drop(columns=['Unnamed: 0', 'name', 'full_name', 'nationality', 'place_of_birth', 'player_agent', 'contract_expires', 'joined_club', 'outfitter'])
data = data.fillna(method='ffill').fillna(method='bfill')

for column in ['position', 'foot', 'club', 'league']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

X = data.drop(columns=['price'])
y = data['price']

# Taking a sample of the dataset to speed up the process
data_sample = data.sample(frac=0.2, random_state=42)
X_sample = data_sample.drop(columns=['price'])
y_sample = data_sample['price']

X_train_sample, X_temp_sample, y_train_sample, y_temp_sample = train_test_split(X_sample, y_sample, test_size=0.4, random_state=42)
X_val_sample, X_test_sample, y_val_sample, y_test_sample = train_test_split(X_temp_sample, y_temp_sample, test_size=0.5, random_state=42)

# Function to plot learning curves
def plot_learning_curves(model, X, y, title=""):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1, 10))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.plot(train_sizes, val_mean, label="Validation score", color="red")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("R^2 Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot predictions vs actual values
def plot_predictions_vs_actual(model, X_val, y_val, title=""):
    y_pred = model.predict(X_val)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Model configurations for adjustment
configurations_sample = {
    'Original': RandomForestRegressor(n_estimators=100, random_state=42),
    'Estimators=50': RandomForestRegressor(n_estimators=50, random_state=42),
    'Max Depth=5': RandomForestRegressor(max_depth=5, random_state=42),
    'No Bootstrap': RandomForestRegressor(bootstrap=False, random_state=42)
}

# Explanations for each model configuration
explanations = {
    'Original': {
        'bias': 'Bajo',
        'variance': 'Media',
        'fit': 'Fit'
    },
    'Estimators=50': {
        'bias': 'Bajo',
        'variance': 'Media',
        'fit': 'Fit'
    },
    'Max Depth=5': {
        'bias': 'Medio',
        'variance': 'Baja',
        'fit': 'Underfit'
    },
    'No Bootstrap': {
        'bias': 'Bajo',
        'variance': 'Alta',
        'fit': 'Fit con alta varianza'
    }
}

# Training models, plotting learning curves and predictions vs actuals on the sampled dataset
for name, model in configurations_sample.items():
    # Train the model on the sampled dataset
    model.fit(X_train_sample, y_train_sample)
    
    # Plot learning curves
    plot_learning_curves(model, X_train_sample, y_train_sample, f'Learning Curve for {name} (Sampled Data)')
    print(f"Diagnóstico de Sesgo (Bias): {explanations[name]['bias']}")
    print(f"Diagnóstico de Varianza: {explanations[name]['variance']}")
    
    # Plot predictions vs actuals
    plot_predictions_vs_actual(model, X_val_sample, y_val_sample, f'Predictions vs Actual for {name} (Sampled Data)')
    print(f"Diagnóstico del Ajuste del Modelo: {explanations[name]['fit']}")
