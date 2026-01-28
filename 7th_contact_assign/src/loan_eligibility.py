import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay

class LoanEligibiltyPredictor:
    def __init__(self, train_path= None, test_path = None, random_state= 42):
        self.random_state = random_state
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None


# identify colums
        self.numerical_columns = ["Loan_Amount_Term", "Credit_History", "ApplicantIncome", "CoapplicantIncome","LoanAmount","Dependents"]
        self.categorial_columns = ["Gender", "Married", "Education", "Self_Employed","Property_Area"]

    def load_data(self):
        if self.train_path:
            self.train_df = pd.read_csv(self.train_path)
        if self.test_path:
            self.test_df = pd.read_csv(self.test_path)

# the 3+ i replaced it and made it a numeric figure 
        self.train_df["Dependents"] = self.train_df["Dependents"].replace("3+", 3).astype(float)
        self.test_df["Dependents"] = self.test_df["Dependents"].replace("3+", 3).astype(float)

        return self
    
    def preprocess(self, target_column="Loan_Status"):
        if self.train_df is None:
            raise ValueError("Training data is loaded")

        X = self.train_df.drop("Loan_Status", axis=1)
        y = self.train_df["Loan_Status"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.random_state, stratify=y)

    def build_processor(self):
        self.preprocessor = ColumnTransformer(
            transformers = [
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), self.numerical_columns),
            
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), self.categorial_columns)
            ]
        )
        return self
    
    def init_models(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(kernel='linear', probability=True, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
        }
        return self

    def train_models(self):
        if not self.models:
            self.init_models()
        if self.preprocessor is None:
            self.build_processor()
        
#  models and name respectively
        for name, model in self.models.items():
            print(f"Training: {name}")

            pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("classifier", model)
            ])

            pipeline.fit(self.X_train, self.y_train)

            y_pred = pipeline.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, pos_label='Y')  # Assuming 'Y' is positive class
            recall = recall_score(self.y_test, y_pred, pos_label='Y')
            f1 = f1_score(self.y_test, y_pred, pos_label='Y')

            #store the values
            self.results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }

        
        #just fo check the best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = pipeline
                self.best_model_name = name

        return self

    def print_results(self):
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name} with Accuracy: {self.best_score:.4f}")

        return self

    def save_model(self):
        joblib.dump(self.results, r"C:\Users\HomePC\python_intermediate_class_2\7th_contact_assign\model\loan_eligibility.joblib")
        return self
    
    def visualize_model(self):
        if self.best_model is None:
            raise ValueError("There is no best model")

        best_data = self.results[self.best_model_name]
        y_pred = best_data['predictions']

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Rejected (N)', 'Approved (Y)']
        )
        
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {self.best_model_name}\nAccuracy: {best_data["accuracy"]:.2%}', 
                 fontsize=14, fontweight='bold', pad=20)

        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')

        plt.show()
    
if __name__ == "__main__":
    loan_predictor = LoanEligibiltyPredictor(
        test_path= r"C:\Users\HomePC\python_intermediate_class_2\7th_contact_assign\data\test_Y3wMUE5_7gLdaTN.csv",
        train_path= r"C:\Users\HomePC\python_intermediate_class_2\7th_contact_assign\data\train_u6lujuX_CVtuZ9i.csv"
    )

    try:
        loan_predictor.load_data()\
                    .preprocess()\
                    .build_processor()\
                    .init_models()\
                    .train_models()\
                    .print_results()\
                    .visualize_model()\
                    .save_model()
    except Exception as e:
        print(f"Error: {e}")
        print("Running step-by-step for debugging...")
        loan_predictor.load_data()
        loan_predictor.preprocess()
        loan_predictor.build_processor()
        loan_predictor.init_models()
        loan_predictor.train_models()
        loan_predictor.print_results()
        loan_predictor.visualize_model()
        loan_predictor.save_model()  
    


