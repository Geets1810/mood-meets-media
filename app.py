import streamlit as st
import pandas as pd

# Load data
mood_logs = pd.read_csv("data/processed/enriched_mood_logs.csv")
media = pd.read_csv("data/processed/netflix_enriched_metadata_hybrid.csv")

# Unique moods
mood_options = sorted(media["mood_tag_standard"].dropna().unique().tolist())

# Sidebar Inputs
st.sidebar.title("🎭 Mood Meets Media")
user_mood = st.sidebar.selectbox("Select your mood", mood_options)

genre_filter = st.sidebar.text_input("Filter by genre (optional)")
media_type_filter = st.sidebar.selectbox("Filter by media type", ["", "Movie", "TV Show", "Music", "Podcast", "ASMR", "Vlog"])
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# Recommendation logic
def recommend_with_filters(mood, df, genre=None, media_type=None, top_n=5):
    filtered = df[df["mood_tag_standard"] == mood]
    if genre:
        filtered = filtered[filtered["genre"].str.contains(genre, case=False, na=False)]
    if media_type:
        filtered = filtered[filtered["type"].str.lower() == media_type.lower()]
    return filtered.sample(n=min(top_n, len(filtered)), random_state=42) if not filtered.empty else pd.DataFrame()

# Header
st.title("🎯 Mood-Based Media Recommendations")

# Get Recommendations
recommendations = recommend_with_filters(user_mood, media, genre_filter, media_type_filter, top_n)

if not recommendations.empty:
    for _, row in recommendations.iterrows():
        st.subheader(row["title"])
        st.write(f"**Type**: {row['type']}  |  **Genre**: {row['genre']}  |  **Mood**: {row['mood_tag_standard']}")
        st.write(f"📄 {row['tags']}")
        st.markdown("---")
else:
    st.warning("No recommendations found. Try adjusting the filters.")
