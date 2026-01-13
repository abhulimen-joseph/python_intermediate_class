import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

train_df = pd.read_csv(r"C:\Users\HomePC\python intermediate class\3rd_contact_assign\7th_contact_assign\data\train_u6lujuX_CVtuZ9i.csv")
test_df = pd.read_csv(r"C:\Users\HomePC\python intermediate class\3rd_contact_assign\7th_contact_assign\data\test_Y3wMUE5_7gLdaTN.csv")

# identify colums
numerical_columns = ["Loan_Amount_Term", "Credit_History", "ApplicantIncome", "CoapplicantIncome","LoanAmount","Dependents"]
classification_columns = ["Gender", "Married", "Education", "Self_Employed","Property_Area"]

# the 3+ i replaced it and made it a numeric figure 
train_df["Dependents"] = train_df["Dependents"].replace("3+", 3).astype(float)
test_df["Dependents"] = test_df["Dependents"].replace("3+", 3).astype(float)

x_train = train_df.drop("Loan_Status", axis=1)
y_train = train_df["Loan_Status"]


preprocess = ColumnTransformer(
    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_columns),
        ("class", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), classification_columns)
    ]
)

# logistic pipeline
logistic_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=300))

])

logistic_pipeline.fit(x_train, y_train)
joblib.dump(logistic_pipeline, r"C:\Users\HomePC\python intermediate class\3rd_contact_assign\7th_contact_assign\model\loan_eligibility.joblib")

test_prediction = logistic_pipeline.predict(test_df)

