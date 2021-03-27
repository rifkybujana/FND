import streamlit as st
import Predict

from Preprocess import *
from Scraper import Scrap

st.write('Fake News Detection AI')

user_input  = st.text_area("Put article url or the full text")
button      = st.button("enter")
language    = st.sidebar.selectbox("Language", ['Bahasa', 'English'])

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
    return Predict.LoadModel(Predict.relative_path + language.lower())

if button:
    text = ""

    with st.spinner("reading article..."):
        if user_input:
            if user_input[:4] == 'http':
                text = Scrap(args.url, args.lang)
            else:
                text = user_input

    if text:
        with st.spinner("Loading Model..."):
            model = load_model()

        preprocess = GetObject(language)

        with st.spinner("Predicting..."):
            prediction = Predict.Predict(model, preprocess, text)

        if prediction >= 0.5:
            activation = round((prediction - 0.5) * 200)
            st.write("Fake [{}%]".format(activation))
        else:
            activation = round((0.5 - prediction) * 200)
            st.write("Valid [{}%]".format(activation))