from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        temperature=0
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Connection failed: {e}")
