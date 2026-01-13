from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

#load model
model = joblib.load("/3rd_contact_assign/5th_contact_session/models")

@app.get("/")
def home():
    return {"message": "Linear Regression API working "}

@app.post("/predict")
def predict():
    arr = np.array(data).reshape(1, -1)
    #make prediction
    pred = model.predict(arr)
    return {"prediction": float(pred[0])}
