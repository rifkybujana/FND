import streamlit as st
import re
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from Scraper import Scrap


model_checkpoint = "Rifky/IndoBERT-FakeNews"


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, fast=True)
    return Trainer(model=model), tokenizer


st.write('# Fake News Detection AI')

with st.spinner("Loading Model..."):
    model, tokenizer = load_model()

user_input = st.text_area("Put article url or the full text", help="the text you want to analyze", height=200)
submit = st.button("submit")

if submit:
    last_time = time.time()

    text = ""

    with st.spinner("Reading Article..."):
        if user_input:
            if user_input[:4] == 'http':
                text = Scrap(user_input)
            else:
                text = user_input

    if text:
        user_input = re.sub(r'\n', ' ', user_input)

        with st.spinner("Computing..."):
            user_input = tokenizer(user_input, max_length=512, truncation=True)
            result = model.predict([user_input])[0][0]
            print (f'\nresult: {result}')

            st.markdown(f"<small>Compute Finished in {time.time() - last_time} seconds</small>", unsafe_allow_html=True)

            if (result[0] >= result[1]):
                st.success("Prediction: Valid")
            else:
                st.error("Prediction: Hoax")
