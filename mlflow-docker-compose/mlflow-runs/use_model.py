import mlflow
import pandas as pd
import numpy as np
import mlflow.pyfunc
from clean_data import *

mlflow.set_tracking_uri("http://localhost:5004")

# If model are not registered, register the model
# result = mlflow.register_model(
#     "runs:/dbd7c762e78f4925bd071b1042b98047/sklearn-model", "sk-learn-random-forest-reg-model"
# )

data = clean_data()

# Load model as a PyFuncModel.  
model_name = "RandomForestRegressor"
# stage = "Staging"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

# Predict on a Pandas DataFrame.
predictions = model.predict(data)

# Compare head of predictions .
print("Example of predictions (10 records):")
for i in range(10):
    print(f"Predicted: {predictions[i]}, Actual: {data.iloc[i]['duration']}")