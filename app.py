import requests
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer as tk
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import load_model

from LogisticRegression import LogisticRegression
from Preprocess import Tokenizer, Encoder

from bs4 import BeautifulSoup


st.write("""# Fake News Detections AI""")

text    = st.text_area("Enter the full article or the link")
submit  = st.button("Enter")


st.sidebar.write("""## Settings""")

language    = st.sidebar.selectbox('language', ['indonesian', 'english'])
model_type  = st.sidebar.selectbox('Machine Learning Model', ['CRNN', 'Logistic Regression'])

st.sidebar.markdown("[Github Page](https://github.com/rifkybujana/FND)")


def Classify(text, language, model):
    with st.spinner('Tokenizing...'):
        tokenizer = Tokenizer(language)
        data = tokenizer.Tokenize(text)

    with st.spinner('Predicting...'):
        if model == 'Logistic Regression':    
            model   = pickle.load(open('Data/Model/Logistic Regression/' + language + '.pkl', 'rb'))
            encoder = pickle.load(open('Data/Encoder/' + language + '.pkl', 'rb'))

            data = encoder.OneHot(data)
            data = np.array(data).reshape(1, len(data))

            prediction = model.predict_prob(data)

        else:    
            model   = load_model('Data/Model/CRNN/' + language + '.h5')
            encoder = pickle.load(open('Data/Tokenizer/' + language + '.pkl', 'rb'))

            data = ' '.join(data)
            data = encoder.texts_to_sequences(np.array([data, ]))
            
            if language == 'indonesian':
                data = pad_sequences(data, maxlen=100)
            else:
                data = pad_sequences(data, maxlen=300)

            prediction = model.predict(data)[0]

    return prediction[0]


if submit:
    data = None

    with st.spinner("Reading Data..."):
        if text:
            if text[:4].lower() == 'http':
                try:
                    url  = requests.get(text)
                    soup = BeautifulSoup(url.content, 'html5lib')

                    article = []
                    for i in soup.findAll('p'):
                        article.append(i.text)

                    article = ' '.join(article)
                    if article:
                        data = article
                    else:
                        st.write("Cant Process the article in this link")
                except Exception as e:
                    st.write(e)
            else:
                data = text

    if data:
        prediction = Classify(data, language, model_type)

        if prediction >= 0.5:
            how_sure = round((prediction - 0.5) * 200, 2)

            if how_sure >= 50:
                st.write("""### i {}% sure its fake""".format(str(how_sure)))
            else:
                st.write("""### im not sure about this, but i think its fake....""")
        else:
            how_sure = round((0.5 - prediction) * 200, 2)

            if how_sure >= 50:
                st.write("""### i {}% sure its valid""".format(str(how_sure)))
            else:
                st.write("""### hmm... im not sure about this, but i think its valid""")


st.text("")
st.text("")
st.warning("""
#### WARNING!
this machine learning model is still a prototype, it may give you wrong prediction.
""")

st.text("")
with st.beta_expander('More Information About this app'):
    st.write("""
    This project is the result of our learning from **AI For Youth** for more than 6 months. 
    This project was created because of our concern about the enormous impact of the very rapid spread of hoax news in Indonesia.
    although this application is still very far from perfect, we hope this application can be useful for anyone.
    and also this application can introduce to many people about artificial intelligence that can really help human life and not vice versa.
    \nAuthor\t: [Rifky Bujana Bisri](mailto:rifkybujanabisri@gmail.com), [Aikyo Dzaki Aroef](mailto:aikyodzakiaroef@gmail.com)
    \nLicense\t: [GPL-3.0 License](https://github.com/rifkybujana/FND/blob/main/LICENSE)
    \nYour can find more information about this app and the source code in the [github repository](https://github.com/rifkybujana/FND)
    """)

    statistic = pd.read_csv('Data/Statistic/overall.csv')
    
    plot, menu = st.beta_columns(2)
    
    lang = menu.selectbox('Language Data', ['Indonesian', 'English'])
    data = menu.radio('Data to show', ['dataset', 'accuracy'])
    
    dataset_data = None
    columns_data = None
    columns_colr = None
    if data == 'dataset':
        dataset_data = statistic[lang][:3].values
        columns_data = np.array(['Total', 'Training', 'Testing'])
        columns_colr = ['r', 'g', 'b']
    else:
        dataset_data = statistic[lang][3:].values
        columns_data = statistic["Parameter"][3:].values
        columns_colr = ['r', 'r', 'b', 'b']
    
    fig, ax = plt.subplots()
    bar = ax.bar(columns_data, dataset_data.tolist(), 0.25, color=columns_colr)
    ax.set_xticklabels(columns_data, rotation=90)
    plot.pyplot(plt)
