import re
import string

import streamlit as st
from keras.src.layers import TextVectorization
from textblob import TextBlob
import pandas as pd
import altair as alt
import tensorflow as tf
import tensorflow_datasets as tfds


# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_resource
def load_model():
    # return tf.keras.models.load_model('models/imdb/output/imdb_conv_1d_word2vec.h5')
    return tf.keras.models.load_model('models/imdb/output/text_sentiment_analysis.h5')


with st.spinner('Model is being loaded..'):
    model = load_model()


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


def vectorizer_test(review):
    return vectorize_layer(review)


VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300


def standardization(input_data):
    """
    Input: raw reviews
    output: standardized reviews
    """
    lowercase = tf.strings.lower(input_data)
    no_tag = tf.strings.regex_replace(lowercase, "<[^>]+>", "")  # take of html tags
    # output = tf.strings.regex_replace(no_tag, "[%s]" % re.escape(string.punctuation), "")

    return no_tag


vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)


def vectorizer_test(review):
    return vectorize_layer(review)


@st.cache_resource
def build_dict():
    # Load the imdb reviews dataset
    train_ds, val_ds, test_ds = tfds.load('imdb_reviews', split=['train', 'test[:50%]', 'test[50%:]'],
                                          as_supervised=True, download=True)

    # Get the training data
    training_data = train_ds.map(lambda x, y: x)

    # Adapt the vectorize_layer to the training data
    vectorize_layer.adapt(training_data)

    return vectorize_layer


def main():
    model = load_model()
    # vectorize_layer = build_dict()

    train_ds, val_ds, test_ds = tfds.load('imdb_reviews', split=['train', 'test[:50%]', 'test[50%:]'],
                                          as_supervised=True)

    # Get the training data
    training_data = train_ds.map(lambda x, y: x)

    # Adapt the vectorize_layer to the training data
    vectorize_layer.adapt(training_data)

    # st.title("Analyze with your own text")
    # st.subheader("Analyze with your own text")

    menu = ["Analyze with your own text", "Batch file sentiment analysis."]
    choice = st.sidebar.selectbox("", menu)

    if choice == "Analyze with your own text":
        st.markdown("<h3>Text Sentiment Analyzer</h3>", unsafe_allow_html=True)
        st.info("Test with movie reviews")
        # st.info("Analyze with your own text")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Text here👇", "Tvery good start, but movie started becoming interesting at some "
                                                  "point and fortunately at some point it started becoming much more "
                                                  "fun, though there was too much background noise, so in all i liked "
                                                  "this movie", help="Analyze with your own text")
            submit_button = st.form_submit_button(label='Analyze the review.')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:
            # Test model
            test_reviews = [raw_text]

            # Convert the list of test reviews to a tensor
            test_reviews_tensor = tf.convert_to_tensor(test_reviews, dtype=tf.string)

            # Vectorize the input text
            test_reviews_vectorized = vectorizer_test(test_reviews_tensor)

            # Make predictions
            predictions = model.predict(test_reviews_vectorized)

            # # Display predictions
            # for review, prediction in zip(test_reviews, predictions):
            #     sentiment = "Positive" if prediction > 0.5 else "Negative"
            #     st.write(f"Review: {review}\nSentiment: {sentiment}\n")

            prediction = predictions[0][0]
            st.info("Results")

            # calculate sentiment predict percent based on prediction

            sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😒"  # ? Natural

            # Emoji
            if prediction >= 0.5:
                # display result with table
                # st.markdown("Sentiment:: Positive 😊 ")
                # st.markdown(f"Confidence: {round(prediction * 100, 2)} percent.")
                # st.markdown(f"Prediction: {prediction}")
                st.dataframe(pd.DataFrame({"Review": [raw_text], "Sentiment": [sentiment], "Prediction": [prediction]}))
                st.balloons()
            else:
                # st.markdown("Sentiment:: Negative 😒 ")
                # st.markdown(f"Confidence: {round(prediction * 100, 2)} percent.")
                # st.markdown(f"Prediction: {prediction}")
                st.dataframe(pd.DataFrame({"Review": [raw_text], "Sentiment": [sentiment], "Prediction": [prediction]}))
                st.snow()

            # with col1:
            #     prediction = predictions[0][0]
            #     st.info("Results")
            #     sentiment = TextBlob(raw_text).sentiment
            #     st.write(sentiment)
            #
            #     # Emoji
            #
            #     if prediction >= 0.5:
            #         st.markdown("Sentiment:: Positive :smiley: 😊 ")
            #         st.balloons()
            #     else:
            #         st.markdown("Sentiment:: Negative :angry: 😒 ")
            #         st.snow()
            # else:
            #     st.markdown("Sentiment:: Neutral 😐 ")

            #     # Dataframe
            #     result_df = convert_to_df(sentiment)
            #     st.dataframe(result_df)
            #
            #     # Visualization
            #     c = alt.Chart(result_df).mark_bar().encode(
            #         x='metric',
            #         y='value',
            #         color='metric')
            #     st.altair_chart(c, use_container_width=True)
            #
            # with col2:
            #     st.info("Token Sentiment")
            #
            #     token_sentiments = analyze_token_sentiment(raw_text)
            #     st.write(token_sentiments)
            #


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
