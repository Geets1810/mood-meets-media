import pandas as pd
import requests
import time

API_KEY = "b5235f3c9ce6844199a5bf0ebb4682e0"
SEARCH_URL = "https://api.themoviedb.org/3/search/multi"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Load your dataset
df = pd.read_csv("cleaned_media_metadata.csv")
df["poster_url"] = ""

for i, row in df.iterrows():
    title = str(row.get("title", "")).strip()
    if not title or title == "nan":
        print(f"Skipping row {i} — missing title")
        continue

    params = {
        "api_key": API_KEY,
        "query": title,
        "include_adult": False
    }

    try:
        response = requests.get(SEARCH_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            for result in data["results"]:
                if result.get("poster_path"):
                    df.at[i, "poster_url"] = IMAGE_BASE_URL + result["poster_path"]
                    print(f"[{i}] ✅ Found poster for: {title}")
                    break
            else:
                print(f"[{i}] ❌ No poster found for: {title}")
        else:
            print(f"[{i}] ❌ No results for: {title}")

    except Exception as e:
        print(f"[{i}] 🔥 Error for '{title}': {e}")

    time.sleep(0.25)

df.to_csv("cleaned_media_metadata_with_posters.csv", index=False)
print("✅ Done! Posters saved.")
