"""
Streamlit dashboard for Mood Meets Media
"""

import streamlit as st

st.title("🎭 Mood Meets Media")
st.markdown("Get personalized content recommendations based on how you feel.")

mood = st.selectbox("Select your current mood", ["Happy", "Anxious", "Tired", "Excited", "Neutral"])

if st.button("Get Recommendations"):
    st.write(f"Showing suggestions for: **{mood}**")
    # TODO: Connect to recommendation engine
