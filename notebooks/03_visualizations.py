import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
df = pd.read_csv("data/cleaned_songs.csv")
from textblob import TextBlob

def get_mood_from_lyrics(row):
    try:
        blob = TextBlob(str(row["lyrics"]))
        polarity = blob.sentiment.polarity
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

df["Better_Mood"] = df.apply(get_mood_from_lyrics, axis=1)
print("Moods generated!")

# Chart 1: Mood Distribution
plt.figure(figsize=(8, 5))
colors = ["red", "steelblue", "green", "pink", "gray"]
df["Better_Mood"].value_counts().plot(kind="bar", color=colors)
plt.title("Music Mood Distribution")
plt.xlabel("Mood")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("notebooks/chart1_mood_distribution.png")
plt.show()
print("Chart 1 saved!")

# Chart 2: Valence by Mood
plt.figure(figsize=(8, 5))
sns.boxplot(x="Better_Mood", y="valence", data=df, palette="Set2")
plt.title("Valence by Mood")
plt.xlabel("Mood")
plt.ylabel("Valence")
plt.tight_layout()
plt.savefig("notebooks/chart2_valence_by_mood.png")
plt.show()
print("Chart 2 saved!")

# Chart 3: Energy by Mood
plt.figure(figsize=(8, 5))
sns.boxplot(x="Better_Mood", y="energy", data=df, palette="Set3")
plt.title("Energy by Mood")
plt.xlabel("Mood")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("notebooks/chart3_energy_by_mood.png")
plt.show()
print("Chart 3 saved!")

# Chart 4: Happy Word Cloud
happy_lyrics = " ".join(df[df["Better_Mood"] == "Happy"]["lyrics"].astype(str))
wc = WordCloud(width=800, height=400,
               background_color="white",
               colormap="YlOrRd").generate(happy_lyrics)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Happy Songs Word Cloud")
plt.tight_layout()
plt.savefig("notebooks/chart4_happy_wordcloud.png")
plt.show()
print("Chart 4 saved!")

print("\nAll charts saved!")