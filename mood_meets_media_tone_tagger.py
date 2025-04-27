
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load media metadata
media_df = pd.read_csv("cleaned_media_metadata_with_posters.csv")

# Load the GoEmotions model
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Load tone vocabulary mapping
tone_df = pd.read_csv("tone_vocabulary_mapping.csv")
emotion_to_tone = {}
for _, row in tone_df.iterrows():
    emotions = [e.strip() for e in row["Mapped Emotions (GoEmotions)"].split(',')]
    for emotion in emotions:
        emotion_to_tone.setdefault(emotion.lower(), []).append(row["Tone Tag"])

goemotions_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def predict_emotions(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].numpy()
    predicted_emotions = [goemotions_labels[i] for i, p in enumerate(probs) if p > threshold]
    return predicted_emotions

def map_emotions_to_tones(emotions):
    tones = set()
    for emotion in emotions:
        mapped = emotion_to_tone.get(emotion.lower(), [])
        tones.update(mapped)
    return list(tones)

media_df['predicted_emotions'] = media_df['description'].apply(predict_emotions)
media_df['tone_tags'] = media_df['predicted_emotions'].apply(map_emotions_to_tones)

# Save to new CSV
media_df.to_csv("media_metadata_with_tone_tags.csv", index=False)
print("✅ Process complete. Output saved to 'media_metadata_with_tone_tags.csv'")
