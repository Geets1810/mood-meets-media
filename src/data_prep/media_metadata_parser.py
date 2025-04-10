"""
Parser for content metadata
"""

def load_metadata(filepath):
    """
    Loads and processes media metadata.
    """
    import pandas as pd
    return pd.read_csv(filepath)
