from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

#load data set
df = pd.read_csv()

#training the piepline
x = df.drop("target_column", axis= 1)
y = df["target_column"]

#split dataset
x_train, x_text, y_train, x_text = train_test_split(x, y, test_size= 0.2,random_state=42)

# Building the pipeline
linear_regression_pipeline = Pipeline([
    ("scaler", StandardScaler),
    ("model", LinearRegression)
])

#Train model
Pipeline.fit(x_train, y_train)

#save trained model
joblib.dump("/3rd_contact_assign/5th_contact_session/models/linear_regression_progression.joblib")

print("Model has been trained and saved successfully")

