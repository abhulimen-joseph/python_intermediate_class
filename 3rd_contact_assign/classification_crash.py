from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from pandas.api.types import is_string_dtype
import seaborn as sn
import numpy as np

# Using the titanic data gotten from kaggle, and with you current knowledge of any of the classification models you are familiar with, perform a classification algorithm to determine passengers that are most likely to survive the titanic crash
# Link to Data: Titanic dataset (kaggle.com)
# Uing logisitic regression which is used for probability

# load the dataset
df = pd.read_csv(r"C:\Users\HomePC\python_intermediate_class_2\3rd_contact_assign\TItanic.csv")
# print(df.head())

#keeping features
important_features = ["sex", "age", "sibsp", "parch", "embarked", "class", "fare"]


contain_null = []
for feature in important_features:
    if df[feature].isnull().any():
        contain_null.append(feature)
# print(contain_null)

# machine learning models don't understand text
contain_str = []
for feature in important_features:
    if is_string_dtype(df[feature]):
        contain_str.append(feature)
# print(contain_str)


df_clean = df[important_features + ["survived"]].copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])

df_clean["sex"] = df["sex"].map({"male": 0, "female": 1})
df_clean["embarked"] = LabelEncoder().fit_transform(df_clean["embarked"])
df_clean["class"] = LabelEncoder().fit_transform(df_clean["class"])

# print("Missing value after cleaning")
# print(df_clean.isnull().sum())
# print("Data types")
# print(df_clean.dtypes)

X = df_clean.drop("survived", axis = 1)
y = df_clean["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


pipeline = Pipeline([
    ("Scaler", StandardScaler()),
    ("Model", LogisticRegression(random_state=42, max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)



print("\n✅ MODEL TRAINED SUCCESSFULLY")
print("="*50)
# print(f"Model parameters: {model.get_params()}")
# print(f"Number of iterations needed: {model.n_iter_[0]}")
print(f"\nTraining accuracy: {pipeline.score(X_train, y_train):.4f}")

# testing model ion case of overfitting
test_accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

#make predictions 

y_pred_prob = pipeline.predict_proba(X_test)[:, 1] #survival probabilities
print(f"Survival Probalities: {y_pred_prob}")





# Using the dataset provided, build a model to classify emails as spam or not spam.
# Link to Data:  Spam Mails Dataset (kaggle.com)