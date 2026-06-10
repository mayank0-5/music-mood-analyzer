import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("data/spotify_songs.csv")
print("Dataset loaded!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Keep only useful columns
df = df[["track_name", "track_artist", "lyrics", 
         "playlist_genre", "valence", "energy", 
         "danceability", "tempo"]].copy()

# Drop rows with missing lyrics
df = df.dropna(subset=["lyrics"])
df = df.drop_duplicates()
print("\nAfter cleaning shape:", df.shape)

# Add Mood based on valence and energy
def get_mood(row):
    valence = row["valence"]
    energy = row["energy"]
    if valence > 0.7 and energy > 0.7:
        return "Energetic"
    elif valence > 0.6:
        return "Happy"
    elif valence < 0.3 and energy < 0.4:
        return "Sad"
    elif valence < 0.4 and energy > 0.6:
        return "Angry"
    else:
        return "Romantic"

df["Mood"] = df.apply(get_mood, axis=1)

print("\nMood Distribution:")
print(df["Mood"].value_counts())

# Save cleaned data
df.to_csv("data/cleaned_songs.csv", index=False)
print("\nCleaned data saved!")