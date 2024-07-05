from st_pages import Page, Section, show_pages
import streamlit as st

st.set_page_config(page_title="Tourists Assistant Chatbot", page_icon=":earth_asia:")
 # ⭐️ 🚀
show_pages(
    [
        Page("pages/NLP_Text_Sentiment_Analysis.py", "Text Sentiment Analyzer", "🚀"),
        Page("pages/3_RAG.py", " Vision Sentiment Analysis (Upcoming)", ":robot_face:"),
        Page("pages/2_Fine_Tune.py", " Sound Sentiment Analysis (Upcoming) ", ":robot_face:")
    ]
)

st.switch_page("pages/NLP_Text_Sentiment_Analysis.py")