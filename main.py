import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("UpdatedResumeDataSet.csv")


# =========================
# 2. REMOVE DUPLICATE RESUMES
# =========================

df = df.drop_duplicates(
    subset=["Resume"]
).reset_index(drop=True)


# =========================
# 3. FEATURES AND TARGET
# =========================

X = df["Resume"]
y = df["Category"]


# =========================
# 4. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 5. TF-IDF
# =========================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# 6. TRAIN MODEL
# =========================

model = LinearSVC()

model.fit(X_train_tfidf, y_train)


# =========================
# 7. PREDICTIONS
# =========================

predictions = model.predict(X_test_tfidf)


# =========================
# 8. EVALUATION
# =========================

accuracy = accuracy_score(y_test, predictions)

print("===== RESUME SCREENING MODEL =====")
print(f"Dataset Size: {len(df)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"TF-IDF Features: {X_train_tfidf.shape[1]}")

print("\n===== MODEL =====")
print("LinearSVC")

print("\n===== ACCURACY =====")
print(f"{accuracy:.2%}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)