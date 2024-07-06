from st_pages import Page, Section, show_pages
import streamlit as st

st.set_page_config(page_title="😊️😒Sentiment Analyzer😊️😒", page_icon="🚀", layout="wide")
# ⭐️ 🚀
show_pages(
    [
        Page("pages/sentiment_analysis_imdb.py", "😊️😒Sentiment Analyzer 😊️😒", "🚀"),
        # Page("pages/3_RAG.py", " Vision Sentiment Analysis (Upcoming)", ":robot_face:"),
        # Page("pages/2_Fine_Tune.py", " Sound Sentiment Analysis (Upcoming) ", ":robot_face:")
    ]
)

# st.switch_page("pages/sentiment_analysis_imdb.py")
