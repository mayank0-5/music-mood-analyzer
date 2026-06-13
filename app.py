import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Music Mood Analyzer", layout="wide")
st.title("🎵 Music Mood Analyzer")
st.write("Paste any song lyrics and discover its mood!")

# Load model
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Mood emojis
mood_emojis = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Energetic": "⚡",
    "Romantic": "💕"
}

# Live Analyzer
st.markdown("---")
st.subheader("🎤 Try It Yourself!")
lyrics = st.text_area("Paste song lyrics here:", height=150,
    placeholder="e.g. I'm walking on sunshine, everything feels so right today...")

if st.button("Analyze Mood"):
    if lyrics:
        lyrics_tfidf = tfidf.transform([lyrics])
        prediction = model.predict(lyrics_tfidf)[0]
        emoji = mood_emojis.get(prediction, "🎵")
        
        st.markdown(f"## Mood: {emoji} **{prediction}**")
        
        # Show probabilities
        probs = model.predict_proba(lyrics_tfidf)[0]
        prob_df = pd.DataFrame({
            "Mood": model.classes_,
            "Probability": probs
        }).sort_values("Probability", ascending=False)
        
        st.bar_chart(prob_df.set_index("Mood"))
    else:
        st.warning("Please paste some lyrics first!")

# Dataset Stats
st.markdown("---")
st.subheader("📊 Dataset Overview")

df = pd.read_csv("data/cleaned_songs.csv")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Songs", len(df))
col2.metric("😊 Happy", len(df[df["Mood"] == "Happy"]))
col3.metric("😢 Sad", len(df[df["Mood"] == "Sad"]))
col4.metric("😠 Angry", len(df[df["Mood"] == "Angry"]))
col5.metric("⚡ Energetic", len(df[df["Mood"] == "Energetic"]))

# Mood distribution chart
st.markdown("---")
st.subheader("Mood Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
df["Mood"].value_counts().plot(kind="bar", color=["pink", "green", "red", "steelblue", "gray"], ax=ax)
ax.set_xlabel("Mood")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
st.pyplot(fig)

# Sample songs
st.markdown("---")
st.subheader("🎵 Sample Songs by Mood")
selected_mood = st.selectbox("Select a mood:", df["Mood"].unique())
sample_songs = df[df["Mood"] == selected_mood][["track_name", "track_artist"]].head(10)
st.dataframe(sample_songs.reset_index(drop=True))