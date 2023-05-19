from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
from clean_data import *

# Data path 
data_path = "Test_Functions/data/bangkok_traffy.csv"

# Load data
Traffyticket = clean_data(data_path)

print("Preparing the data to split train/test .....")
y = Traffyticket.pop('duration')
X = Traffyticket


# ## Creating and Preparing the Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

print("Hyperparameter tuning and feature selection .....")

# Define the parameter gridSearch 
param_grid = {
    'criterion':['friedman_mse','squared_error'],
    'max_depth': [2,3,6],
    'min_samples_leaf':[2,5,10],
    'n_estimators':[100,200],
    'random_state': [2023]
}

# Create the model
rf = RandomForestRegressor()

''' BEGIN HYPERPARAMETER TUNING '''
# Create a Grid Search object 
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Train the gridSearch onject on the training data
grid_search.fit(X, y)

# Print the best hyperparameters found by grid search
print("Best hyperparameters: ", grid_search.best_params_)

# Evaluate the performance of the model on the validation set
test_acc = grid_search.score(X, y)
print("Test accuracy: ", test_acc)

model = grid_search.best_estimator_
# Print the best hyperparameters found by grid search
print("Best estimators: ", grid_search.best_estimator_)

# Get the best parameters
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
best_rf = grid_search.best_estimator_

''' END HYPERPARAMETER TUNING '''

''' BEGIN FEATURE SELECTION '''
# Use feature selection to select the most important features
selector = SelectFromModel(best_rf, threshold='median')
selector.fit(X, y)

# Print the selected feature names
selected_features = X.columns[selector.get_support()]
print('Selected features:', list(selected_features))

''' END FEATURE SELECTION '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

# ## Model Training & Evaluation


import os
import warnings
import sys

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)                                                                                                                      
model_name = "RandomForestRegressor"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

mlflow.set_tracking_uri("http://localhost:5004")

with mlflow.start_run():
    # Train the model on the best parameters
    rf_best = RandomForestRegressor(**best_params)
    rf_best.fit(X_train, y_train)

    # Infer the model signature
    y_pred = rf_best.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print(f"{model_name} model with parameters: {best_params}")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    '''{'criterion': 'friedman_mse', 'max_depth': 6, 'min_samples_leaf': 10, 'n_estimators': 100, 'random_state': 2023}'''
    mlflow.log_param("criterion", best_params['criterion'])
    mlflow.log_param("max_depth", best_params['max_depth'])
    mlflow.log_param("min_samples_leaf", best_params['min_samples_leaf'])
    mlflow.log_param("n_estimators", best_params['n_estimators'])
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    print(mlflow.get_tracking_uri())
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(tracking_url_type_store)

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(sk_model=rf_best, artifact_path="sklearn-model", signature=signature, registered_model_name=f"{model_name}")
    else:
        mlflow.sklearn.log_model(rf_best, "model")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report

# Generate predictions for the testing set
y_pred = rf_best.predict(X_test)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Assume y_true and y_pred are the true and predicted labels, respectively
# print(classification_report(y_test, y_pred))

import pandas as pd

# Assume y_true and y_pred are the true and predicted labels, respectively
df = pd.DataFrame({'true': y_test, 'predicted': y_pred})
print(df.head(15))