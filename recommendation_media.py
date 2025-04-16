import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

mood_keywords = {
    'happy': ['joyful', 'fun', 'lighthearted', 'celebration', 'romantic comedy'],
    'sad': ['uplifting', 'hope', 'feel-good', 'inspirational', 'healing'],
    'anxious': ['calm', 'soothing', 'nature', 'slow-paced', 'comforting'],
    'low energy': ['short', 'motivational', 'bright', 'positive', 'animated'],
    'inspired': ['empowering', 'biography', 'success', 'drama', 'achievement'],
    'exhausted': ['lighthearted', 'feel-good', 'short', 'positive', 'fun', 'calm'],
    'excited': ['calm', 'soothing', 'celebration', 'romantic', 'joyful'],
    'neutral': ['thriller', 'haunting', 'scary', 'fun', 'drama', 'uplifting'],
    'disregulated': ['calm', 'soothing', 'slow-paced', 'feel-good', 'empowering', 'inspirational']
}

mood_transitions = {
    'anxious': ['soothed', 'calm', 'hopeful'],
    'sad': ['comforted', 'inspired', 'uplifted'],
    'low energy': ['calm', 'motivated', 'energized'],
    'exhausted': ['soothed', 'lighthearted', 'recharged'],
    'disregulated': ['calm', 'centered', 'empowered'],
    'neutral': ['curious', 'uplifted', 'stimulated'],
    'excited': ['focused', 'joyful', 'grounded'],
    'happy': ['relaxed', 'lighthearted'],
    'inspired': ['confident', 'purposeful', 'calm']
}

emotion_to_mood = {
    'soothed': 'calm',
    'comforted': 'feel-good',
    'recharged': 'lighthearted',
    'hopeful': 'inspired',
    'uplifted': 'happy',
    'stimulated': 'inspired',
    'empowered': 'inspired',
    'centered': 'calm',
    'confident': 'inspired',
    'purposeful': 'inspired',
    'curious': 'neutral',
    'focused': 'inspired',
    'joyful': 'happy',
    'relaxed': 'calm',
    'lighthearted': 'happy',
    'energized': 'inspired',
    'motivated': 'inspired'
}

import re

def is_mood_cancelled(desc, keyword):
    desc = desc.lower()
    keyword = keyword.lower()

    # Break the description into sentences
    sentences = re.split(r'[.!?]', desc)

    # Phrases that reverse emotional tone
    reversers = [
    'dark turn', 'takes a dark turn', 'takes a dark and decidedly dangerous turn',
    'goes wrong', 'turns deadly', 'ends in disaster',
    'horrifying twist', 'unexpectedly', 'turns into a nightmare',
    'not what it seems', 'with deadly consequences', 'becomes terrifying',
    'descends into chaos'
    ]


    for sentence in sentences:
        if keyword in sentence:
            # Only check reversal *within the same sentence*
            for phrase in reversers:
                if phrase in sentence:
                    return True
    return False

def recommend_media(user_mood, media_df, top_n=5):
    mood_text = ' '.join(mood_keywords.get(user_mood, []))

    tfidf = TfidfVectorizer(stop_words='english')
    media_desc_matrix = tfidf.fit_transform(media_df['description'].fillna(''))
    mood_vec = tfidf.transform([mood_text])

    similarity_scores = cosine_similarity(mood_vec, media_desc_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1]  # Sorted from high to low

    # Go through sorted results and apply negation-aware filtering
    filtered_recs = []
    for idx in top_indices:
        row = media_df.iloc[idx]
        desc = str(row['description']).lower()
        valid = True
        for keyword in mood_keywords.get(user_mood, []):
            if keyword.lower() in desc and is_mood_cancelled(desc, keyword):
                valid = False
                break
        if valid:
            filtered_recs.append(row)
        if len(filtered_recs) == top_n:
            break

    return pd.DataFrame(filtered_recs)[['title', 'description', 'genre', 'type', 'poster_url']]
