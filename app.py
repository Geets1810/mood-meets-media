import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
from textblob import TextBlob
from recommendation_media import recommend_media, mood_keywords, mood_transitions,emotion_to_mood,is_mood_cancelled

# Load and clean your media dataset
media_df = pd.read_csv("cleaned_media_metadata_with_posters.csv")

# Clean and normalize poster_url column
def clean_url(val):
    val = str(val).strip()
    return val if val.startswith("http") else ""

media_df["poster_url"] = media_df["poster_url"].apply(clean_url)

# Streamlit app layout
st.set_page_config(page_title="Mood Meets Media", layout="centered")
st.title("🎬 Mood Meets Media")
st.markdown("🌿 *Let your mood guide your next binge...*")
st.markdown("---")

# Optional: Journal-based input
# --- Mood Input Section ---
st.markdown("### 💭 How would you like to choose your mood today?")
mood_input_mode = st.radio("Choose input method:", ["📝 Write how you're feeling", "🎯 Pick from mood list"])

if mood_input_mode == "📝 Write how you're feeling":
    journal_entry = st.text_area("Describe how you feel right now:")

    def detect_mood_from_journal(text):
        if not text.strip():
            return None
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity < -0.3:
            return 'sad'
        elif polarity < 0:
            return 'anxious'
        elif 0 <= polarity < 0.2:
            return 'neutral'
        elif 0.2 <= polarity < 0.5:
            return 'inspired'
        else:
            return 'happy'

    mood = detect_mood_from_journal(journal_entry)

    if mood:
        st.success(f"🧠 Based on your journal, we think you're feeling: **{mood}**")
    else:
        st.warning("Please write a bit more so we can detect your mood.")

else:
    mood = st.selectbox("🧠 Choose a mood", list(mood_keywords.keys()))

# Mood + type filters
#mood = st.selectbox("🧠 What's your mood today?", list(mood_keywords.keys()))
media_type = st.selectbox("🎥 What would you like to watch?", ["Any", "Movie", "TV Show"])

# Filter by type
if media_type != "Any":
    filtered_df = media_df[media_df["type"].str.lower() == media_type.lower()]
else:
    filtered_df = media_df
if not mood:
    st.stop()

# Recommend content
if st.button("Recommend"):
    recs = recommend_media(mood, filtered_df, top_n=5)

    if recs.empty:
        st.warning("😕 Sorry, we couldn’t find any matches. Try another mood or type!")
    else:
        st.subheader("✨ Here's what we think you'll enjoy:")

        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 3])

            # 🖼️ Poster
            with col1:
                image_url = row.get("poster_url", "")
                if image_url:
                    st.image(image_url, width=120)
                else:
                    st.image("https://via.placeholder.com/120x180.png?text=No+Image", width=120)




            # 📖 Details
            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*Genre:* {row['genre']}")
                st.markdown(f"*Description:* {row['description']}")

                # Match mood keywords
                matched_keywords = []
                desc = str(row['description']).lower()
                for keyword in mood_keywords[mood]:
                    if keyword.lower() in desc and not is_mood_cancelled(desc, keyword):
                        matched_keywords.append(keyword)

                if matched_keywords:
                    st.markdown(f"🔍 *Why this?* Matched keywords: `{', '.join(matched_keywords)}`")
                else:
                    st.markdown("🔍 *Why this?* Based on genre and emotional tone.")

            st.markdown("---")


# ----- EMOTIONAL TRANSITION SECTION -----
next_emotions = mood_transitions.get(mood, [])
if next_emotions:
    st.markdown("### 🌱 Want to gently shift your mood?")
    for emotion in next_emotions:
        mapped_mood = emotion_to_mood.get(emotion, 'happy')
        st.markdown(f"**➡️ {emotion.title()}**")

        transition_recs = recommend_media(mapped_mood, filtered_df, top_n=2)
        for _, row in transition_recs.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                image_url = row.get("poster_url", "")
                if image_url.startswith("http"):
                    st.image(image_url, width=100)
                else:
                    st.image("https://via.placeholder.com/120x180.png?text=No+Image", width=100)

            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*{row['genre']}*")

                # 🔍 Match keywords with negation-aware check
                matched_keywords = []
                desc = str(row['description']).lower()
                for keyword in mood_keywords.get(mapped_mood, []):
                    if keyword.lower() in desc and not is_mood_cancelled(desc, keyword):
                        matched_keywords.append(keyword)

                if matched_keywords:
                    st.markdown(f"🔍 *Why this?* Matched keywords: `{', '.join(matched_keywords)}`")
                else:
                    st.markdown(f"*To help you feel:* `{emotion}`")

            st.markdown("---")
