"""
Transform mood logs into usable format
"""

def transform_logs(df):
    """
    Normalize and structure mood logs for analysis.
    """
    df["date"] = pd.to_datetime(df["date"])
    return df
