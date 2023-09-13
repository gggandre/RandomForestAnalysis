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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a function to plot the dataset separation
def plot_dataset_separation(X_train, X_val, X_test):
    """
    The function `plot_dataset_separation` plots the separation of a dataset into training, validation,
    and test sets based on the features 'age' and 'height'.
    :param X_train: X_train is the training dataset, which contains the features (age and height) for
    training the model
    :param X_val: X_val is a pandas DataFrame containing the validation data. It has two columns: 'age'
    and 'height'. The 'age' column represents the age of individuals in the validation set, and the
    'height' column represents their corresponding heights
    :param X_test: X_test is a pandas DataFrame containing the test data. It has two columns: 'age' and
    'height'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train
    axes[0].scatter(X_train['age'], X_train['height'], c='blue', alpha=0.6, label='Training Data')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Height')
    axes[0].set_title('Training Data')
    
    # Validation
    axes[1].scatter(X_val['age'], X_val['height'], c='green', alpha=0.6, label='Validation Data')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Height')
    axes[1].set_title('Validation Data')

    # Test
    axes[2].scatter(X_test['age'], X_test['height'], c='red', alpha=0.6, label='Test Data')
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Height')
    axes[2].set_title('Test Data')

    plt.tight_layout()
    plt.show()

# Call the function to plot the dataset separation
plot_dataset_separation(X_train, X_val, X_test)

# 1. Curves of learning for bias and variance diagnosis
def plot_learning_curves(estimator, X_train, y_train, title):
    """
    The function `plot_learning_curves` plots the learning curves of an estimator using the training and
    validation scores.
    :param estimator: The estimator is the machine learning model that you want to evaluate. It could be
    any scikit-learn estimator object, such as a classifier or a regressor
    :param X_train: X_train is the input features of the training data. It is a matrix or array-like
    object with shape (n_samples, n_features), where n_samples is the number of samples and n_features
    is the number of features for each sample
    :param y_train: The parameter `y_train` represents the target variable or the labels for the
    training data. It is a one-dimensional array or a column vector that contains the true values
    corresponding to each sample in the training dataset
    :param title: The title of the learning curve plot. It is used as the title of the plot
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    plt.figure(figsize=(10, 6))
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='b')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_mean, 'o-', color='b', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='r', label='Validation score')
    plt.title(title)
    plt.xlabel('Number of training samples')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# 2. Graphics of the training and validation scores for the diagnosis of the fit
def plot_predictions_vs_actual(estimator, X_val, y_val, title):
    """
    The code defines a function to plot the predictions versus the actual values for a given estimator,
    and then uses different configurations of a random forest regressor to train models, plot learning
    curves, plot predictions versus actual values, and store performance metrics.
    :param estimator: The `estimator` parameter is the machine learning model that you want to evaluate.
    It should be an instance of a regression model that has been fitted on the training data
    :param X_val: X_val is the validation set of input features. It is a matrix or dataframe containing
    the input features for the validation data
    :param y_val: The variable `y_val` represents the actual target values for the validation set. It is
    a numpy array or pandas Series containing the true values of the target variable for the validation
    data
    :param title: The title parameter is a string that specifies the title of the plot. It is used to
    provide a descriptive title for the plot, indicating what it represents
    """
    y_pred = estimator.predict(X_val)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Model configurations for adjustment
configurations = {
    'Original': RandomForestRegressor(n_estimators=100, random_state=42),
    'Estimators=50': RandomForestRegressor(n_estimators=50, random_state=42),
    'Max Depth=5': RandomForestRegressor(max_depth=5, random_state=42),
    'No Bootstrap': RandomForestRegressor(bootstrap=False, random_state=42)
}

performance_metrics = {}
for name, model in configurations.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Plot learning curves for bias and variance diagnosis
    plot_learning_curves(model, X_train, y_train, f'Learning Curve for {name}')
    
    # Plot predictions vs actual for fit diagnosis
    plot_predictions_vs_actual(model, X_val, y_val, f'Predictions vs Actual for {name}')
    
    # Store performance metrics
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    r2 = r2_score(y_val, y_val_pred)
    performance_metrics[name] = [mae, mse, rmse, r2]

# 3. Comparison of Performance Metrics for Regularization or Parameter Adjustment Techniques
# The code `performance_df = pd.DataFrame(performance_metrics, index=['MAE', 'MSE', 'RMSE', 'R^2'])`
# creates a pandas DataFrame called `performance_df` using the `pd.DataFrame()` function. It takes the
# `performance_metrics` dictionary as input, where the keys are the model configurations and the
# values are lists of performance metrics (MAE, MSE, RMSE, R^2) for each model configuration.
performance_df = pd.DataFrame(performance_metrics, index=['MAE', 'MSE', 'RMSE', 'R^2'])
print(performance_df)
