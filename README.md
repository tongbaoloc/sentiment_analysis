
# Sentiment Analysis System

## Project Overview

This project focuses on creating a sentiment analysis system that classifies input data into positive or negative sentiment categories. The system is designed to analyze text, images(upcoming), and audio(upcoming) to determine sentiment. It leverages Long Short-Term Memory (LSTM) networks and pre-trained word embeddings from Google’s Word2Vec model in text sentiment analysic   . The project is built using TensorFlow and Keras and includes a Streamlit application for easy demonstration and usage.

## System Architecture

### Text Sentiment Analysis
- **LSTM (Long Short-Term Memory)**: Captures temporal dependencies in text data.
- **Word2Vec-Google-News-300**: Utilizes pre-trained word embeddings from Google’s Word2Vec model, which is trained on the Google News dataset with 300 dimensions.

### Image Sentiment Analysis (Upcoming)
- **Pre-trained CNN (Convolutional Neural Network)**: Extracts features from images to classify sentiment.

### Audio Sentiment Analysis (Upcoming)
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Extracts features from audio signals.
- **LSTM**: Analyzes the temporal features extracted from audio data.

## Technology Stack

- **TensorFlow**: An open-source deep learning framework.
- **Keras**: A high-level neural networks API written in Python, running on top of TensorFlow.
- **Google Colab**: Used for building and training the models before integrating them into the Streamlit application.

## Streamlit Application

Streamlit is used to create an interactive web application for demonstrating the sentiment analysis system.

## How to Run the Application

### File Structure
- **app.py**: The main Streamlit application file.
- **model.py**: Contains the implementation of the LSTM model and loading of pre-trained embeddings.
- **utils.py**: Utility functions for preprocessing and feature extraction.
- **requirements.txt**: List of required Python packages.
- **text_sentiment_colab_notebook.ipynb,sound_sentiment_colab_notebook.ipynb,image_sentiment_colab_notebook.ipynb**: Google Colab notebook for building and training the models.

### Using Google Colab

Steps to Build and Train Models in Google Colab:

1. **Open the Colab Notebook**: Open `colab_notebook.ipynb` in Google Colab.
2. **Run the Cells**: Execute the cells in the notebook to build and train the models.
3. **Save the Models**: Save the trained models to your Google Drive or download them to your local machine.
4. **Load the Models in Streamlit**: Move the saved models to your Streamlit project directory and load them in the `model.py` file.

### Running the Streamlit Application

1. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required Python packages.
2. **Start the Application**: Run `streamlit run main.py` in your terminal to start the Streamlit application.
3. **Interact with the Application**: Use the web interface to test the sentiment analysis on text, image, and audio inputs.

### Referencing the Research Paper

This project is based on the following research paper for implementation details and theoretical background:

- **Title**: “Deep Learning for Sentiment Analysis: A Survey”
- **Authors**: Zhang, Lei, Shuai Wang, and Bing Liu.
- **Link**: [Deep Learning for Sentiment Analysis: A Survey](https://link-to-paper.com)
