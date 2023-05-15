# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch

# %%
# Data path 
data_path = "../../Test_Functions/data/bangkok_traffy.csv"

# Load data
Traffyticket = pd.read_csv(data_path)

# %%
Traffyticket.head()

# %%
Traffyticket.info()

# %%
Traffyticket.describe()

# %%
Traffyticket.shape

# %%
print(f"Number of rows: {Traffyticket.shape[0]}")
# Filter to collect ticket which state is 'เสร็จสิ้น'
Traffyticket = Traffyticket[Traffyticket['state'] == 'เสร็จสิ้น']
Traffyticket.head(3)

# Filter to remove nan value
print("Dropping nan value....")
Traffyticket = Traffyticket.dropna()    
print(f"Number of rows: {Traffyticket.shape[0]}")

# %%
Traffyticket.columns

# %%
drop_columns = ['photo', 'photo_after', 'ticket_id', 'coords', 'address', 'comment', 'state', 'last_activity', 'timestamp', 'star', 'subdistrict', 'organization', 'count_reopen']

# Calculate by convert last_activity and timestamp to datetime and calculate to add the duration column
Traffyticket['last_activity'] = pd.to_datetime(Traffyticket['last_activity'])
Traffyticket['timestamp'] = pd.to_datetime(Traffyticket['timestamp'])
Traffyticket['duration'] = Traffyticket['last_activity'] - Traffyticket['timestamp']

# Convert duration to days
Traffyticket['duration'] = Traffyticket['duration'].dt.days

# Show the result
Traffyticket.head(3)

# %%
# Drop the columns
Traffyticket.drop(drop_columns, axis=1, inplace=True)
# Reset index
Traffyticket.reset_index(drop=True, inplace=True)
# Show the result
Traffyticket.head(3)

# %%
print(Traffyticket.info())

# %%
Traffyticket

# %%
def extract_types(df):
    result = []
    for index,row in Traffyticket.iterrows():
        types = row['type'].strip('{}').split(',')
        for t in types: 
            new_row = row.copy()
            new_row['type'] = t.strip()
            result.append(new_row)
    return result

# %%
Traffyticket = pd.DataFrame(extract_types(Traffyticket))

Traffyticket.head(10)

# %% [markdown]
# ## Prepare the data before train the model

# %%
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

nominal_columns = ['type', 'district', 'province']

# Fit the encoder
enc.fit(Traffyticket[nominal_columns])

# Transform the categorical columns to numerical columns
enc_cols = enc.transform(Traffyticket[nominal_columns]).toarray()

# Create the new datafram with the encoded columns
enc_df = pd.DataFrame(enc_cols, columns=enc.get_feature_names_out(nominal_columns))

# Merge the original dataframe with the encoded dataframe
Traffyticket = enc_df.join(Traffyticket)

# Drop the original categorical columns 
Traffyticket.drop(nominal_columns, axis=1, inplace=True)

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer

# # Define which columns should be one-hot encoded
# categorical_cols = ['type', 'district', 'province']

# # Define which columns should be scaled
# numeric_cols = ['count_reopen']

# # Create transformers for each column type
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')
# numeric_transformer = StandardScaler()

# # Combine the transformers into a single preprocessor
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', categorical_transformer, categorical_cols),
#         ('num', numeric_transformer, numeric_cols)
#     ])

# # Fit the preprocessor on the training data
# preprocessor.fit(X_train)

# # Transform the training and testing data
# X_train = preprocessor.transform(X_train)
# X_test = preprocessor.transform(X_test)


# %%
print(f"Shape of the dataframe before dropping: {Traffyticket.shape}")

# Drop NaN value
Traffyticket.dropna(inplace=True)

print(f"Shape of the dataframe after dropping: {Traffyticket.shape}")
# Show the result
Traffyticket.head(10)

# %%
print(Traffyticket.info())

# %% [markdown]
# ## Train Test Split

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils, datasets

def data_loader(valid_size=0.1,
                random_seed=31,
                batch_size=32,
                shuffle=True):

    # load the dataset
    train_dataset = extract_types(Traffyticket)
    valid_dataset = extract_types(Traffyticket)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

# %%
y = Traffyticket.pop('duration')
X = Traffyticket


# %% [markdown]
# ## Creating and Preparing the Model

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

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

# %% [markdown]
# ## Model Training & Evaluation

# %%
import os
import warnings
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

with mlflow.start_run():
    # Train the model on the best parameters
    rf_best = RandomForestRegressor(**best_params)
    rf_best.fit(X_train, y_train)

    y_pred = rf_best.predict(X_test)

    mlflow.set_tracking_uri("http://localhost:5000")

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
    # model_signature = infer_signature(train_x, train_y)
    print(tracking_url_type_store)

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(rf_best, "model", registered_model_name={model_name})
    else:
        mlflow.sklearn.log_model(rf_best, "model")

# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report

# Generate predictions for the testing set
y_pred = rf_best.predict(X_test)

# Calculate the MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R-squared:", r2)

# Assume y_true and y_pred are the true and predicted labels, respectively
# print(classification_report(y_test, y_pred))

# %%
import pandas as pd

# Assume y_true and y_pred are the true and predicted labels, respectively
df = pd.DataFrame({'true': y_test, 'predicted': y_pred})
print(df.head())