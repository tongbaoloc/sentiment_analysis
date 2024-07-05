import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
import tensorflow as tf


# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/text_sentiment_analysis.h5')


with st.spinner('Model is being loaded..'):
    model = load_model()


# Fxn
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df


def analyze_token_sentiment(docx):
    # analyzer = SentimentIntensityAnalyzer()
    # pos_list = []
    # neg_list = []
    # neu_list = []
    # for i in docx.split():
    #     res = analyzer.polarity_scores(i)['compound']
    #     if res > 0.1:
    #         pos_list.append(i)
    #         pos_list.append(res)
    #
    #     elif res <= -0.1:
    #         neg_list.append(i)
    #         neg_list.append(res)
    #     else:
    #         neu_list.append(i)
    #
    # result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    # return result
    return None


def main():
    # st.title("Test with your own text.")
    # st.subheader("Test with your own text.")

    menu = ["Test with your own text.", "Batch file sentiment analysis."]
    choice = st.sidebar.selectbox("", menu)

    if choice == "Test with your own text.":
        st.info("Test with your own text.")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Text here👇")
            submit_button = st.form_submit_button(label='Classify Text 🔍')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji

                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: 😊 ")
                    st.balloons()
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: 😒 ")
                    st.snow()
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Token Sentiment")

                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)







    elif choice == "Batch file sentiment analysis.":
        # st.info("Batch file sentiment analysis.")
        ui_batch_file_rendering()


def ui_batch_file_rendering(special_internal_function=None):
    st.markdown("<h3>Text Sentiment Analyzer</h3>", unsafe_allow_html=True)

    st.info("Batch files sentiment analysis.")

    st.caption("To perform sentiment analysis on multiple text files, upload the files below.")

    with open("upload_files/fine_tuning_data/fine_tuning_ask_and_answer_template.xlsx", "rb") as aa_template_file:
        aa_template_byte = aa_template_file.read()

    st.download_button(
        label="Download the template",
        data=aa_template_byte,
        file_name="fine_tuning_ask_and_answer_template.xlsx"
    )

    st.write("Q&A example example:")

    aa_template_file = pd.read_excel("upload_files/fine_tuning_data/fine_tuning_ask_and_answer_template.xlsx")
    st.write(aa_template_file)


if __name__ == '__main__':
    main()
