import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

#load dataset
df = pd.read_csv(r"C:\Users\HomePC\python_intermediate_class_2\4th_contact_assign\kc_house_data.csv")
print(df.head())

# data preprocessing
column_nan = []
column = list(df)
contain_string = list(df)
# print(column)
def check_null(column): # checking for null
    for col in column:
        if df[col].isnull().any():
            column_nan.append(col)
    print(column_nan)

# print(df.columns)

def check_string(contain_string):
    for st in contain_string:
        if df[st].dtype == "object":
            contain_string.append(st)
    print("These columns contain strings:\n ", contain_string)

check_string(contain_string)


# the feature and the target need to be seperated 
X = df.drop("price", axis = 1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# scaler.fit()
# X_train_scaler = scaler.transform(X_train)
# X_test_scaler = scaler.transform(X_test)

pipeline = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42))
        ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        (Ridge(random_state=42))
        ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        (Lasso(random_state=42))
        ]),

    "DecisionTree": Pipeline([
        ("scaler", StandardScaler()),
        (DecisionTreeRegressor(random_state=42))
        ])
}
results = []

for name, model_pipeline in pipeline.items():
    print("Model:", name)
    model_pipeline.fit(X_train, y_train)

    # make predictions  
    y_train_predict = model_pipeline.predict(X_train)
    y_test_predict = model_pipeline.predict(y_test)

    #make evaluation 
    train_mae = mean_absolute_error(y_train, y_train_predict)
    test_mae = mean_absolute_error(y_test, y_test_predict)

    train_r2_score = r2_score(y_train, y_train_predict)
    test_r2_score = r2_score(y_test, y_test_predict)

    #cross validartion score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs= -1) 
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results.append({
        "Model": name,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2_score": train_r2_score,
        "test_r2_score": test_r2_score,
        "cv_r2_mean": cv_mean,
        "cv_std": cv_scores
    })

results_df = pd.DataFrame(results)
print(results_df.head())

