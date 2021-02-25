import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import base64
import nltk
import pickle

from numpy import log, dot, e
from io import StringIO
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#############################################################################################
###################################### SIDEBAR ##############################################
#############################################################################################

st.sidebar.write("""
### Settings
""")

model = st.sidebar.selectbox('Model', ['Logistic Regression','Naive Bayse','Recurrent Neural Networks'])
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
######################################   MODEL  #############################################
#############################################################################################

class LogisticRegressions:
    def __init__(self, lr=0.05, epochs=100, intercept=True):
        self.lr = lr
        self.epochs = epochs
        self.intercept = intercept
        self.bias = 0
    
    def addIntercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.intercept:
            X = self.addIntercept(X)
        
        self.loss = []
        
        # weights initialization
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.epochs):
            z = np.dot(X, self.weight)
            h = self.sigmoid(z + self.bias)
            
            gradient = np.dot(X.T, (h - y)) / y.size
            bGradient = np.sum(h - y) / y.size
            
            self.weight -= self.lr * gradient
            self.bias -= self.lr * bGradient
            
            self.loss.append(self.cost(h, y))
    
    def predict_prob(self, X):
        if self.intercept:
            X = self.addIntercept(X)
    
        return self.sigmoid(np.dot(X, self.weight) + self.bias)
    
    def predict(self, X):
        return self.predict_prob(X) >= 0.5

id_model = pickle.load(open('indo_model.pkl', 'rb'))
en_model = pickle.load(open('eng_model.pkl', 'rb'))

id_vocab = pd.read_csv('indo_vocab.csv', header=None).values
id_vocab = id_vocab.reshape(len(id_vocab)).tolist()

en_vocab = pd.read_csv('eng_vocab.csv', header=None).values
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
    text = ' '.join(re.sub(r'[^\w+ |_]', ' ', text).split()).lower()

    if language == 'english':
        text = lemmatizer.lemmatize(text)
    else:
        text = stemmer.stem(text)

    text = re.sub(r'\d+','num', text)
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords and len(w) > 2]
    text = list(dict.fromkeys(text))

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
        if i == "num":
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