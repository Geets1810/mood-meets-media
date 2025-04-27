# --- Import libraries ---
import streamlit as st
import pandas as pd
import nltk
from transformers import pipeline
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from recommendation_media import recommend_media, mood_keywords, mood_transitions, emotion_to_mood, is_mood_cancelled

# Load HuggingFace emotion classification model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

#Detect Aggregated Mood
def detect_aggregated_mood(text):
    if not text.strip():
        return None

    sentences = sent_tokenize(text)
    mood_list = []

    for sentence in sentences:
        mood = detect_mood_from_journal(sentence)
        if mood:
            mood_list.append(mood)

    if mood_list:
        # Pick the most common mood (majority voting)
        final_mood = max(set(mood_list), key=mood_list.count)
        return final_mood
    else:
        return None

# --- Define helper functions ---
# Detect Mood from Journal
def detect_mood_from_journal(text):
    if not text.strip():
        return None

    # Use HuggingFace emotion classifier
    results = emotion_classifier(text)

    if results:
        results = results[0]  # get the list of emotions with scores
        # Sort by highest score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        # Pick the top emotion
        top_emotion = sorted_results[0]['label'].lower()
        return top_emotion
    else:
        return None

#Adjust mood based on context
def adjust_mood_based_on_context(journal_text, detected_mood):
    journal_text = journal_text.lower()
    exhaustion_keywords = ["exhausted", "tired", "drained", "long day", "burnt out", "fatigued", "unwind", "need to unwind", "overwhelmed"]

    if any(keyword in journal_text for keyword in exhaustion_keywords):
        if detected_mood in ["anxious", "sad"]:  # lowercase because HuggingFace outputs lowercase
            return "tired"
    return detected_mood

#Redifened Detected Mood

def refine_detected_mood(journal_text, detected_mood):
    journal_text = journal_text.lower()

    # Keywords to help refine
    hopeful_keywords = ["hope", "hopeful", "grateful", "thankful", "looking forward", "optimistic", "excited for"]
    stress_keywords = ["stress", "stressed", "overwhelmed", "burnt out", "tired", "exhausted", "anxious"]

    # If model predicts Neutral but journal has hopeful words
    if detected_mood == "neutral":
        if any(keyword in journal_text for keyword in hopeful_keywords):
            return "hopeful"

    # If model predicts Fear but journal matches stress context
    if detected_mood == "fear":
        if any(keyword in journal_text for keyword in stress_keywords):
            return "anxious"

    # If model predicts Surprise but journal has grateful/hopeful words
    if detected_mood == "surprise":
        if any(keyword in journal_text for keyword in hopeful_keywords):
            return "hopeful"

    # Else return original detected mood
    return detected_mood


#clean_url
def clean_url(val):
    val = str(val).strip()
    return val if val.startswith("http") else ""

# --- Load and clean dataset ---
media_df = pd.read_csv("cleaned_media_metadata_with_posters.csv")
media_df["poster_url"] = media_df["poster_url"].apply(clean_url)

# --- Streamlit app layout ---
st.set_page_config(page_title="Mood Meets Media", layout="centered")
st.title("🎬 Mood Meets Media")
st.markdown("🌿 *Let your mood guide your next binge...*")
st.markdown("---")

# --- Mood input section ---
st.markdown("### 💭 How would you like to choose your mood today?")
mood_input_mode = st.radio("Choose input method:", ["📝 Write how you're feeling", "🎯 Pick from mood list"])

if mood_input_mode == "📝 Write how you're feeling":
    journal_entry = st.text_area("Describe how you feel right now:")

    if journal_entry:
        detected_mood = detect_aggregated_mood(journal_entry)
        if detected_mood:
            mood = adjust_mood_based_on_context(journal_entry, detected_mood)
            st.success(f"🧠 Based on your journal, we think you're feeling: **{mood}**")
        else:
            st.warning("Please write a bit more so we can detect your mood.")
    else:
        mood = None

else:
    mood = st.selectbox("🧠 Choose a mood", list(mood_keywords.keys()))

# --- Media type filter ---
media_type = st.selectbox("🎥 What would you like to watch?", ["Any", "Movie", "TV Show"])

# --- Filter media dataset ---
if media_type != "Any":
    filtered_df = media_df[media_df["type"].str.lower() == media_type.lower()]
else:
    filtered_df = media_df

# --- Stop if mood is None ---
if not mood:
    st.stop()

# --- Recommend content ---
if st.button("Recommend"):
    recs = recommend_media(mood, filtered_df, top_n=5)

    if recs.empty:
        st.warning("😕 Sorry, we couldn’t find any matches. Try another mood or type!")
    else:
        st.subheader("✨ Here's what we think you'll enjoy:")

        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 3])

            with col1:
                image_url = row.get("poster_url", "")
                if image_url:
                    st.image(image_url, width=120)
                else:
                    st.image("https://via.placeholder.com/120x180.png?text=No+Image", width=120)

            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*Genre:* {row['genre']}")
                st.markdown(f"*Description:* {row['description']}")

# --- Emotional Transition section ---
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
                if image_url:
                    st.image(image_url, width=100)
                else:
                    st.image("https://via.placeholder.com/120x180.png?text=No+Image", width=100)

            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*{row['genre']}*")
