import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_songs.csv")
print("Data loaded! Shape:", df.shape)

# Label mood from lyrics using TextBlob + valence
def get_mood_from_lyrics(row):
    try:
        blob = TextBlob(str(row["lyrics"]))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        valence = row["valence"]
        energy = row["energy"]

        if polarity > 0.2 and energy > 0.7:
            return "Energetic"
        elif polarity > 0.2 and valence > 0.6:
            return "Happy"
        elif polarity < -0.1 and energy > 0.6:
            return "Angry"
        elif polarity < -0.1 and valence < 0.4:
            return "Sad"
        else:
            return "Romantic"
    except:
        return "Romantic"

print("\nGenerating better mood labels...")
df["Better_Mood"] = df.apply(get_mood_from_lyrics, axis=1)

print("\nNew Mood Distribution:")
print(df["Better_Mood"].value_counts())

# Features and target
X = df["lyrics"]
y = df["Better_Mood"]

# TF-IDF
print("\nApplying TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = tfidf.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Train Model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training done!")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("\nModel saved!")

# Test
sample = "I am so happy today, everything is wonderful and bright!"
prediction = model.predict(tfidf.transform([sample]))[0]
print(f"\nSample mood: {prediction}")