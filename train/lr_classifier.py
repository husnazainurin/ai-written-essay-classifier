import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle
import textwrap

def preprocesses_text(text):
    '''This method will lower case all letters, replace special characters with space 
    and replace consecutive spaces or line breaks with a space. '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Import dataset
data = pd.read_csv("dataset/reviews_class.csv")

# Strip the reviews (remove spacing at the front and back of the text)
data.columns = data.columns.str.strip()

# Split predictors from target class
x = data["text"].str.strip()
y = data["label"]

# Apply tf-idf using English stopwords and maximum of 2 n-grams
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=preprocesses_text,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("logreg", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])


# Perform model evaluation using cross-validation method
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    x,
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

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)

print("FINAL TEST SET EVALUATION")
print("==============================================================")
print("Accuracy:", accuracy_score(y_test, predictions))
print("--------------------------------------------------------------")
print(classification_report(y_test, predictions))
print("--------------------------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the model
with open('model/lr_model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)


# New input testing. But this will be done in GUI app interface so I'm commenting it out. Feel free to uncomment this!
# print("==============================================================\n")
# new_text = input("\n  In about 50 words, tell us about the best TV show you have ever watch,\n"
#                  "  and why? Explain a little, like who's you favorite character, \n"
#                  "  or favorite scenes, and etc.: \n")

# wrapped_text = textwrap.fill(new_text, width=70)  # wrap at 70 chars
# print("\nYour input:")
# print(wrapped_text)

# prediction = pipeline.predict([new_text])

# print("\nYour text is predicted to be by LR: ")
# print("AI-generated" if prediction[0] == 1 else "Human-written")







