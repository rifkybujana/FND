from msilib import sequence
import streamlit as st
import numpy as np
import re
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from Scraper import Scrap

st.set_page_config(layout="wide")

model_checkpoint = "Rifky/FND"
label = {0: "valid", 1: "fake"}


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, fast=True)
    return Trainer(model=model), tokenizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


input_column, reference_column = st.columns(2, gap="medium")
input_column.write('# Fake News Detection AI')

with st.spinner("Loading Model..."):
    model, tokenizer = load_model()

user_input = input_column.text_input("Article url")
submit = input_column.button("submit")


if submit:
    last_time = time.time()
    with st.spinner("Reading Article..."):
        if user_input:
            if user_input[:4] == 'http':
                text = Scrap(user_input)
            else:
                text = user_input

    if text:
        text = re.sub(r'\n', ' ', text)

        with st.spinner("Computing..."):
            text = text.split()
            text_len = len(text)

            sequences = []
            for i in range(text_len // 512):
                sequences.append(" ".join(text[i * 512: (i + 1) * 512]))
            sequences.append(" ".join(text[text_len - (text_len % 512) : text_len]))
            sequences = [tokenizer(i, max_length=512, truncation=True, padding="max_length") for i in sequences]

            predictions = model.predict(sequences)[0]
            result = [
                np.sum([sigmoid(i[0]) for i in predictions]) / len(predictions), 
                np.sum([sigmoid(i[1]) for i in predictions]) / len(predictions)
            ]
                
            print (f'\nresult: {result}')
            input_column.markdown(f"<small>Compute Finished in {int(time.time() - last_time)} seconds</small>", unsafe_allow_html=True)
            prediction = np.argmax(result, axis=-1)
            input_column.success(f"This news is {label[prediction]}.")
            st.text(f"{int(result[prediction]*100)}% confidence")
            input_column.progress(result[prediction])
