import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import os

from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from Scraper import Scrap

st.set_page_config(layout="wide")

model_checkpoint = "Rifky/indobert-hoax-classification"
base_model_checkpoint = "indobenchmark/indobert-base-p1"
data_checkpoint = "Rifky/indonesian-hoax-news"
label = {0: "valid", 1: "fake"}

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    base_model = SentenceTransformer(base_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, fast=True)
    data = load_dataset(data_checkpoint, split="train")
    return model, base_model, tokenizer, data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

input_column, reference_column = st.columns(2)
input_column.write('# Fake News Detection AI')

with st.spinner("Loading Model..."):
    model, base_model, tokenizer, data = load_model()

user_input = input_column.text_input("Article url")
submit = input_column.button("submit")


if submit:
    last_time = time.time()
    with st.spinner("Reading Article..."):
        scrap = Scrap(user_input)
        title, text = scrap.title, scrap.text

    if text:
        text = re.sub(r'\n', ' ', text)

        with st.spinner("Computing..."):
            token = text.split()
            text_len = len(token)

            sequences = []
            for i in range(text_len // 512):
                sequences.append(" ".join(token[i * 512: (i + 1) * 512]))
            sequences.append(" ".join(token[text_len - (text_len % 512) : text_len]))
            sequences = tokenizer(sequences, max_length=512, truncation=True, padding="max_length", return_tensors='pt')

            predictions = model(**sequences)[0].detach().numpy()
            result = [
                np.sum([sigmoid(i[0]) for i in predictions]) / len(predictions), 
                np.sum([sigmoid(i[1]) for i in predictions]) / len(predictions)
            ]
                
            print (f'\nresult: {result}')
            
            title_embeddings = base_model.encode(title)
            similarity_score = cosine_similarity(
                [title_embeddings],
                data["embeddings"]
            ).flatten()
            sorted = np.argsort(similarity_score)[::-1].tolist()
            
            input_column.markdown(f"<small>Compute Finished in {int(time.time() - last_time)} seconds</small>", unsafe_allow_html=True)
            prediction = np.argmax(result, axis=-1)
            input_column.success(f"This news is {label[prediction]}.")
            input_column.text(f"{int(result[prediction]*100)}% confidence")
            input_column.progress(result[prediction])
            
            for i in sorted[:5]:
                reference_column.write(f"""
                <small>{data["url"][i].split("/")[2]}</small>
                <a href={data["url"][i]}><h5>{data["title"][i]}</h5></a>
                """, unsafe_allow_html=True)
