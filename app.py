import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import base64
import nltk
import pickle

from LogisticRegression import LogisticRegressions
from io import StringIO
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from joblib import Parallel, delayed

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

id_model = pickle.load(open('Data/indo_model.pkl', 'rb'))
en_model = pickle.load(open('Data/eng_model.pkl', 'rb'))

id_vocab = pd.read_csv('Data/indo_vocab.csv', header=None).values
id_vocab = id_vocab.reshape(len(id_vocab)).tolist()

en_vocab = pd.read_csv('Data/eng_vocab.csv', header=None).values
en_vocab = en_vocab.reshape(len(en_vocab)).tolist()

stopwords = set(stopwords.words(language))
lemmatizer = WordNetLemmatizer()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

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

# preprocess the text
def Preprocess(text):
    text = text.lower()
    text = text.split()
    text = ' '.join(Parallel(n_jobs=-1)(delayed(re.sub)(r'[^A-Za-z0-9]+', '', w) for w in text))

    if language == 'english':
        text = text.split()
        text = Parallel(n_jobs=-1)(delayed(lemmatizer.lemmatize)(i) for i in text)
    else:
        text = stemmer.stem(text).split()

    text = ' '.join(Parallel(n_jobs=-1)(delayed(re.sub)(r'\d+', ' <num> ', w) for w in text)).split()
    text = list(dict.fromkeys(text))
    text = [w for w in text if not w in stopwords and len(w) > 2]

    return text

# Process all text in file
def ProcessData(file):
    with st.spinner("Processing Data...."):
        progress = st.progress(0)
        for i in range(len(file)):
            file[i] = Preprocess(file[i])
            progress.progress((i + 1) / len(file))

        progress.empty()
        return file

# turn text into bow
def Vectorize(text):
    vocab = []
    if language == 'english':
        vocab = en_vocab
    else:
        vocab = id_vocab

    result = [0] * len(vocab)
    for i in text:
        if i == "<num>":
            result[vocab.index("num")] = 1
        else:
            try:
                result[vocab.index(i)] = 1
            except:
                result[vocab.index("unk")] = 1
    
    return result

# vectorize all text in file
def VectorizeData(file):
    with st.spinner("Vectorizing Data...."):
        progress = st.progress(0)
        for i in range(len(file)):
            file[i] = Vectorize(file[i])
            progress.progress((i + 1) / len(file))

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
            data.append(text)

    # saving the original data
    original = data.copy()

    # Preprocess the data
    Preprocessed_data = ProcessData(data)

    # Vectorize the data
    VectorizedData = np.asarray(VectorizeData(Preprocessed_data))

    # Reshape Data
    if len(VectorizedData.shape) < 2:
        VectorizedData = VectorizedData.reshape(1, len(VectorizedData))

    # Predict this news is fake or not
    model = LogisticRegressions()
    if language == 'english':
        model = en_model
    else:
        model = id_model
    
    Prediction = model.predict(VectorizedData)

    Result = pd.DataFrame({
        'Text' : original,
        'Preprocessed Data' : Preprocessed_data,
        'Prediction' : Prediction
    })
    Result['Prediction'] = Result['Prediction'].replace([True, False], ["Hoax", "Valid"])
    st.dataframe(Result)

    st.markdown(GetCSVDownloadLink(Result, "Result.csv"), unsafe_allow_html=True)