import mlflow
import pandas as pd
import numpy as np
import mlflow.pyfunc
from clean_data import *

mlflow.set_tracking_uri("http://localhost:5004")

result = mlflow.register_model(
    "runs:/e6b832258f7d433686ed7a4dfa042cdc/sklearn-model", "sk-learn-random-forest-reg-model"
)

data_path = "Test_Functions/data/bangkok_traffy.csv"

data = clean_data(data_path)

# Load model as a PyFuncModel.  
model_name = "sk-learn-random-forest-reg-model"
# stage = "Staging"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

# Predict on a Pandas DataFrame.
predictions = model.predict(data)
print(predictions)