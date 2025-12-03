import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
df = pd.read_csv("target")

# features and target
x = df.drop["target column"]
y = df["other column"]

# Split again
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

# model
model = joblib.load("/3rd_contact_assign/5th_contact_session/models/linear_progression_pipeline.joblib")

#predict
y_pred = model.predict(x_test)

#evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation result:")
print("Mean squared Ealuation:", mse)
print("R2 score: ", r2)

