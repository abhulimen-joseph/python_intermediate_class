import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from evaluate_model import evaluate_model


boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# data cleaning
df = df.apply(pd.to_numeric, errors = "coerce")
df = df.fillna(df.mean())

# categorize into x and y
x = df.drop("MEDV", axis=1)
y = df["MEDV"]

#testing and training 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.astype(float)
y_train = y_train.astype(float)

# ridge pipeline build
ridge_regression_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1))
])

# train model
ridge_regression_pipeline.fit(x_train, y_train)

#save trained model
joblib.dump(ridge_regression_pipeline, r"C:\Users\HomePC\python intermediate class\3rd_contact_assign\4th_contact_assign\models\ridge_regression_model.joblib")

print("model has been created and saved successfully")

# evaluate model
results = evaluate_model(ridge_regression_pipeline, x_test, y_test)
print("Ridge results:", results)