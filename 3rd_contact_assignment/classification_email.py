import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import nltk
from  nltk.corpus import stopwords

nltk.download("stopwords")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

df = pd.read_csv(r"")
# print(df.head())
# print(f"shape: {df.shape}")
# print(f"Missing_values: {df.isnull().sum()}")

#pre processing data
df["Cleaned_text"] = df["text"].apply(preprocess_text)
# print(df['Cleaned_text'])

# print(f"\nUnique labels: {df["label"].unique()}")
if df["label"].dtype == "object":
    df["numeric_label"] = df["label"].map({"ham": 0, "spam": 1})
    y = df["numeric_label"]
else:
    print("label is not an object")

X = df["Cleaned_text"]

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

pipeline = Pipeline([
    ("Tfid", TfidfVectorizer(
        max_features=5000,
        min_df = 5,
        max_df = 0.8,
        stop_words= "english",
        ngram_range= (1, 2)
    )),
    ("classifier", MultinomialNB(alpha=0.1))
])

#train pipeline
pipeline.fit(X_train, y_train)

#make preedictions
predictions = pipeline.predict(X_test)

#Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy_score: {accuracy}")
print(classification_report(y_test, predictions, target_names=["ham", "spam"] if "ham" in df["label"].unique() else["class 0", "class 1"] ))

joblib.dump(pipeline, "spam_email.pkl")
print("Model saved into spam_joblib.pkl")