"""
Mood Classifier using journal entries
"""

def classify_mood(entry):
    """
    Classifies mood from a text journal entry.
    """
    # Dummy logic
    if "happy" in entry.lower():
        return "Happy"
    elif "anxious" in entry.lower():
        return "Anxious"
    else:
        return "Neutral"
