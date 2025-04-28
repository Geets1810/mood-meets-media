# --- Import libraries ---
import os

import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# Set nltk data path
nltk.data.path.append('./nltk_data')


# --- Safely ensure required NLTK data is available ---

# List of resources with their correct paths
nltk_dependencies = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'vader_lexicon': 'sentiment/vader_lexicon'
}

# Check and download if not found
for dependency, path in nltk_dependencies.items():
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(dependency)
        except Exception as e:
            print(f"Error downloading {dependency}: {e}")

from recommendation_media import recommend_media, mood_keywords, mood_transitions, emotion_to_mood, is_mood_cancelled

# Load GoEmotions model
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

emotion_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# --- Helper Functions ---

def clean_url(val):
    val = str(val).strip()
    return val if val.startswith("http") else ""

def detect_mood_from_journal(text):
    if not text.strip():
        return None
    results = emotion_classifier(text)
    if results:
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        return sorted_results[0]['label'].lower()
    else:
        return None

def detect_aggregated_mood(text):
    if not text.strip():
        return None
    sentences = sent_tokenize(text)
    mood_list = [detect_mood_from_journal(sentence) for sentence in sentences if detect_mood_from_journal(sentence)]
    if mood_list:
        final_mood = max(set(mood_list), key=mood_list.count)
        return final_mood
    else:
        return None

def refine_detected_mood(journal_text, detected_mood):
    journal_text = journal_text.lower()
    hopeful_keywords = ["hope", "hopeful", "grateful", "thankful", "looking forward", "optimistic", "excited for"]
    stress_keywords = ["stress", "stressed", "overwhelmed", "burnt out", "tired", "exhausted", "anxious"]

    if detected_mood == "neutral" and any(keyword in journal_text for keyword in hopeful_keywords):
        return "hopeful"
    if detected_mood == "fear" and any(keyword in journal_text for keyword in stress_keywords):
        return "anxious"
    if detected_mood == "surprise" and any(keyword in journal_text for keyword in hopeful_keywords):
        return "hopeful"
    return detected_mood

def remap_detected_emotion_to_mood(detected_emotion):
    detected_emotion = detected_emotion.lower()

    uplifted_emotions = ["joyful", "hopeful", "proud", "optimistic", "excited", "enthusiastic", "energetic", "playful", "amused"]
    low_energy_emotions = ["sad", "lonely", "vulnerable", "grieving", "powerless", "empty", "isolated", "depressed", "abandoned", "ashamed"]
    stressed_emotions = ["anxious", "overwhelmed", "panicked", "worried", "nervous", "insecure", "scared", "fearful"]
    irritated_emotions = ["frustrated", "annoyed", "angry", "hostile", "resentful", "bitter", "mad"]
    connected_emotions = ["caring", "affectionate", "compassionate", "trusting", "loving", "appreciative"]
    calm_emotions = ["content", "relaxed", "peaceful", "balanced", "satisfied", "secure", "at ease"]
    curious_emotions = ["curious", "inquisitive", "amazed", "awestruck", "intrigued", "fascinated"]
    conflicted_emotions = ["confused", "disillusioned", "withdrawn", "apprehensive", "indifferent", "unsure"]

    if detected_emotion in uplifted_emotions:
        return "Uplifted / Inspired"
    elif detected_emotion in low_energy_emotions:
        return "Low Energy / Needs Comfort"
    elif detected_emotion in stressed_emotions:
        return "Stressed / Overwhelmed"
    elif detected_emotion in irritated_emotions:
        return "Irritated / Frustrated"
    elif detected_emotion in connected_emotions:
        return "Connected / Heartwarming"
    elif detected_emotion in calm_emotions:
        return "Calm / Peaceful / Content"
    elif detected_emotion in curious_emotions:
        return "Curious / Stimulated"
    elif detected_emotion in conflicted_emotions:
        return "Conflicted / Uncertain"
    else:
        return "Curious / Stimulated"

def adjust_mood_based_on_context(journal_text, detected_mood):
    journal_text = journal_text.lower()
    exhaustion_keywords = ["exhausted", "tired", "drained", "long day", "burnt out", "fatigued", "unwind", "need to unwind", "overwhelmed"]

    if any(keyword in journal_text for keyword in exhaustion_keywords):
        if detected_mood in ["anxious", "sad"]:
            return "Tired"
    return detected_mood

def preclassify_complex_emotions(journal_text):
    journal_text = journal_text.lower()

    complex_emotion_mappings = {
        "Stressed / Overwhelmed": ["exhausted", "drained", "burnt out", "depleted", "fatigued", "wiped out"],
        "Low Energy / Needs Comfort": ["bittersweet", "happy and sad", "nostalgic pain", "empty", "numb", "disconnected", "emotionally flat"],
        "Uplifted / Inspired": ["empowered", "stronger today", "resilient", "unstoppable", "overwhelmed with happiness", "overwhelmed but grateful", "inspired", "motivated", "purpose-driven"],
        "Conflicted / Uncertain": ["disillusioned", "jaded", "lost hope"],
        "Curious / Stimulated": ["in awe", "amazed beyond words", "overwhelmed by beauty"]
    }

    for mood_group, keywords in complex_emotion_mappings.items():
        if any(keyword in journal_text for keyword in keywords):
            return mood_group

    return None

def generate_emotional_reflection(mood, journal_text=None):
    system_prompt = (
        "You are an emotionally supportive assistant helping users based on their current mood. "
        "Write a short 1-2 sentence warm reflection that acknowledges their emotional state gently. "
        "The tone should be kind, validating, and encouraging."
    )

    user_prompt = f"My detected mood is: {mood}."
    if journal_text:
        user_prompt += f" Here’s some context from my journal: {journal_text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.6,
        max_tokens=80
    )

    reflection = response.choices[0].message.content
    return reflection


# Set your API key safely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load and clean dataset ---
media_df = pd.read_csv("cleaned_media_metadata_with_posters.csv")
media_df["poster_url"] = media_df["poster_url"].apply(clean_url)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Mood Meets Media", layout="centered")
st.title("🎬 Mood Meets Media")
st.markdown("🌿 *Let your mood guide your next binge...*")
st.markdown("---")

# --- Mood Input Section ---
st.markdown("### 💭 How would you like to choose your mood today?")
mood_input_mode = st.radio("Choose input method:", ["📝 Write how you're feeling", "🎯 Pick from mood list"])

# Mood input mode
if mood_input_mode == "📝 Write how you're feeling":
    journal_entry = st.text_area("Describe how you feel right now:")

    if journal_entry.strip():  # ✅ Make sure journal entry is not empty
        # First check complex emotions
        mood = preclassify_complex_emotions(journal_entry)

        if not mood:
            # If no complex emotion detected, use model-based detection
            raw_detected_emotion = detect_aggregated_mood(journal_entry)
            refined_emotion = refine_detected_mood(journal_entry, raw_detected_emotion)
            stable_mood = remap_detected_emotion_to_mood(refined_emotion)
            final_mood = adjust_mood_based_on_context(journal_entry, stable_mood)
            mood = final_mood

        if mood:
                    # New - Generate emotional reflection
            try:
                reflection_text = generate_emotional_reflection(mood, journal_entry)
                if reflection_text:
                    st.markdown(f"💬 *{reflection_text}*")
            except Exception as e:
                st.error(f"Reflection generation failed: {e}")

        else:
            st.warning("⚡ We couldn't detect your mood clearly. Please try writing a bit more!")

    else:
        mood = None

# Else, fallback to dropdown
else:
    mood = st.selectbox("🧠 Choose a mood", list(mood_keywords.keys()))


# --- Media Type Filter ---
media_type = st.selectbox("🎥 What would you like to watch?", ["Any", "Movie", "TV Show"])

if media_type != "Any":
    filtered_df = media_df[media_df["type"].str.lower() == media_type.lower()]
else:
    filtered_df = media_df

if not mood:
    st.stop()


# --- Recommend Content ---
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
                st.image(image_url if image_url else "https://via.placeholder.com/120x180.png?text=No+Image", width=120)
            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*Genre:* {row['genre']}")
                st.markdown(f"*Description:* {row['description']}")

# --- Emotional Transition Section ---
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
                st.image(image_url if image_url else "https://via.placeholder.com/120x180.png?text=No+Image", width=100)
            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*{row['genre']}*")
