# 🎭 Mood Meets Media

**Mood Meets Media** is an interactive AI-powered wellness app that recommends music, movies, podcasts, and more based on your mood. It uses emotion detection, journaling insights, and public media datasets to generate personalized recommendations.

### 🚀 Features
- Detect your mood from mood logs or journal entries
- Explore recommendations by genre, type (movie, music, etc.), or mood
- Powered by Hugging Face emotion models and Streamlit
- Embed the app directly into your Notion or website

### 📦 Folder Structure
```
mood-meets-media/
├── app.py
├── data/
│   ├── processed/
│   │   ├── enriched_mood_logs.csv
│   │   └── merged_media_metadata.csv
├── requirements.txt
├── README.md
```

### 🧠 Technologies Used
- Python
- Streamlit
- Hugging Face Transformers
- Pandas
- Jupyter

### 📈 How to Run
1. Clone the repo:
```bash
git clone https://github.com/yourusername/mood-meets-media.git
cd mood-meets-media
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

### 🌍 Deploy
Deploy this app on [Streamlit Community Cloud](https://streamlit.io/cloud) and embed it in Notion for a public-facing, interactive wellness tool!

---
Made with 💫 by Geethanjali Vivekanandan
