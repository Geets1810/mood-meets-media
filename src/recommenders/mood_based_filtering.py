"""
Mood-based filtering logic
"""

def filter_by_mood(content_df, current_mood):
    """
    Filter content based on user mood.
    """
    return content_df[content_df["mood_tag"] == current_mood]
