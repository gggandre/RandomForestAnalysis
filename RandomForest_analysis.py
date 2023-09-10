import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv('top5_leagues_player.csv')
data = data.drop(columns=['Unnamed: 0', 'name', 'full_name', 'place_of_birth', 'shirt_nr', 'player_agent', 'contract_expires', 'joined_club'])

# Fill missing values and encode categorical columns
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
        data[column] = le.fit_transform(data[column])
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Split data into training, validation, and test sets
X = data.drop(columns=['price'])
y = data['price']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Train the original model and get metrics
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)

# Train the tuned model and get metrics
rf_tuned = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_tuned.fit(X_train, y_train)
y_val_pred_tuned = rf_tuned.predict(X_val)
y_test_pred_tuned = rf_tuned.predict(X_test)

# Visualize the results
plt.figure(figsize=(15, 6))

# Original model
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red')
plt.title('Original Model: Actual vs Predicted Prices (Validation Set)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# Tuned model
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_val_pred_tuned, alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red')
plt.title('Tuned Model: Actual vs Predicted Prices (Validation Set)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.tight_layout()
plt.show()
