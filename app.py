import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import base64
import nltk
import pickle
import requests

from bs4 import BeautifulSoup
from LogisticRegression import LogisticRegressions
from Preprocess import Tokenizer, Encoder
from io import StringIO

#############################################################################################
###################################### SIDEBAR ##############################################
#############################################################################################

st.sidebar.write("""
### Settings
""")

language = st.sidebar.selectbox('language', ['english', 'indonesian'])

#############################################################################################
###################################### MAIN UI ##############################################
#############################################################################################

title = st.empty()
title.write("""
# Fake News Detection AI
## This App Will Predict Whether a News is **Fake** or Not!
""")

text_input = st.empty()
text = text_input.text_area("Enter the article title or the link")

file_input = st.empty()
file = file_input.file_uploader("Or, you can upload a list of article", type=['txt'])

submit_button = st.empty()
submit = submit_button.button("Enter")

#############################################################################################
##################################### LOAD MODEL  ###########################################
#############################################################################################

id_model = pickle.load(open('Data/id_logistic.pkl', 'rb'))
en_model = pickle.load(open('Data/en_logistic.pkl', 'rb'))

tokenizer = Tokenizer(language)
id_encoder = pickle.load(open('Data/id_encoder.pkl', 'rb'))
en_encoder = pickle.load(open('Data/en_encoder.pkl', 'rb'))

#############################################################################################
###################################### FUNCTION #############################################
#############################################################################################

# create a download link to download the data as csv file
def GetCSVDownloadLink(data, filename):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False, header=None)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'

    return href


# read .txt file
def ReadFile(file):
    return StringIO(file.getvalue().decode("utf-8"))


# Process all text in file
def ProcessData(file):
    with st.spinner("Processing Data...."):
        progress = st.progress(0)
        for i in range(len(file)):
            file[i] = tokenizer.Tokenize(file[i])
            progress.progress((i + 1) / len(file))

        time.sleep(0.5)
        progress.empty()
        return file


# vectorize all text in file
def VectorizeData(file):
    if language == 'english':
        encoder = en_encoder
    else:
        encoder = id_encoder

    with st.spinner("Vectorizing Data...."):
        progress = st.progress(0)
        for i in range(len(file)):
            file[i] = encoder.OneHot(file[i])
            progress.progress((i + 1) / len(file))

        time.sleep(0.5)
        progress.empty()
        return file

# if the submit button pressed
if submit:
    # Read Data
    with st.spinner("Reading Data..."):
        data = []

        # process file if there's any file uploaded
        if not file is None:
            data += ReadFile(file)

        # process text if there's any text written
        if len(text) > 0:
            if text[:4].lower() == 'http':
                try:
                    url = requests.get(text)
                    soup = BeautifulSoup(url.content, 'html5lib')

                    article = []
                    for i in soup.findAll('p'):
                        article.append(i.text)

                    article = ' '.join(article)
                    
                    if article:
                        data.append(article)
                except Exception as e:
                    st.write(e)
            else:
                data.append(text)

    if data:
        # saving the original data
        original = data.copy()

        # Preprocess the data
        Preprocessed_data = ProcessData(data)

        # Vectorize the data
        VectorizedData = np.asarray(VectorizeData(Preprocessed_data))

        # Reshape Data into 2D shape
        if len(VectorizedData.shape) < 2:
            VectorizedData = VectorizedData.reshape(1, len(VectorizedData))

        # Predict this news is fake or not
        if language == 'english':
            Prediction = en_model.predict(VectorizedData)
        else:
            Prediction = id_model.predict(VectorizedData)

        # Show Prediction
        Result = pd.DataFrame({
            'Text' : original,
            'Preprocessed Data' : Preprocessed_data,
            'Prediction' : Prediction
        })
        Result['Prediction'] = Result['Prediction'].replace([True, False], ["Hoax", "Valid"])
        st.dataframe(Result)

        st.markdown(GetCSVDownloadLink(Result, "Result.csv"), unsafe_allow_html=True)