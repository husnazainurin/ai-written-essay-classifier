import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from  sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import pickle
import sys
from pathlib import Path
import textwrap


sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.preprocess import preprocesses_text


# Import dataset
data = pd.read_csv("dataset/reviews_class.csv")

# Strip the reviews (remove spacing at the front and back of the text)
data.columns = data.columns.str.strip()

# Split predictors from target class
X = data["text"].str.strip()
y = data["label"]

# Apply tf-idf using English stopwords and maximum of 2 n-grams
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=preprocesses_text,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("svm", SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        random_state=42
    ))
])

# Model evaluation using cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


# Display result
print("==============================================================")
print("CROSS-VALIDATION RESULTS")
print("==============================================================")
print("Accuracy for each fold:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("==============================================================\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("FINAL TEST SET EVALUATION")
print("==============================================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("--------------------------------------------------------------")
print(classification_report(y_test, y_pred))
print("--------------------------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("==============================================================\n")

# Save the model
with open('model/svm_model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)


# New input testing. But this will be done in GUI app interface so I'm commenting it out. Feel free to uncomment this!
# new_text = input(
#     "\nIn about 50 words, tell us about the best TV show you have ever watched,\n"
#     "and why? Explain a little, like your favorite character or scene:\n\n"
# )

# wrapped_text = textwrap.fill(new_text, width=70)
# print("\nYour input:")
# print(wrapped_text)

# prediction = pipeline.predict([new_text])

# print("\nPrediction Result by SVM:")
# print("AI-generated" if prediction[0] == 1 else "Human-written")
# print()

