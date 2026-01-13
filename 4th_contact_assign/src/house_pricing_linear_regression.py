import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from evaluate_model import evaluate_model

boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame
# print(df.head())
df = df.apply(pd.to_numeric, errors = "coerce")
df = df.fillna(df.mean())

# categorize into x and y 
x = df.drop("MEDV", axis=1)
y = df["MEDV"]

#testing and training 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

x_train = x_train.astype(float)
y_train = y_train.astype(float)

#Linear regression pipeline build
linear_regression_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# train model
linear_regression_pipeline.fit(x_train, y_train)

# save trained model
joblib.dump(linear_regression_pipeline, r"C:\Users\HomePC\python intermediate class\3rd_contact_assign\4th_contact_assign\models\linear_regression_model.joblib")

print("model has been trained and saved sucessfully")

results = evaluate_model(linear_regression_pipeline, x_test, y_test)
print("Evaluation results:", results)
