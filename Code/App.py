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

model_checkpoint = "Rifky/indobert-hoax-classification"
base_model_checkpoint = "indobenchmark/indobert-base-p1"
data_checkpoint = "Rifky/indonesian-hoax-news"
label = {0: "valid", 1: "fake"}

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    base_model = SentenceTransformer(base_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, fast=True)
    data = load_dataset(data_checkpoint, split="train", download_mode='force_redownload')
    return model, base_model, tokenizer, data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

with st.spinner("Loading Model..."):
    model, base_model, tokenizer, data = load_model()

st.markdown("""<h1 style="text-align:center;">Fake News Detection AI</h1>""", unsafe_allow_html=True)
user_input = st.text_input("Article URL")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    margin: auto;
    display: block;
    width: 100%;
}
</style>""", unsafe_allow_html=True)

submit = st.button("submit")

if submit:
    last_time = time.time()
    with st.spinner("Reading Article..."):
        scrap = Scrap(user_input)

    if scrap:
        title, text = scrap.title, scrap.text
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

            prediction = np.argmax(result, axis=-1)
            if prediction == 0:
                st.markdown(f"""<p style="background-color: rgb(236, 253, 245); 
                color: rgb(6, 95, 70);
                font-size: 20px;
                border-radius: 7px;
                padding-left: 12px;
                padding-top: 15px;
                padding-bottom: 15px;
                line-height: 25px;
                text-align: center;">This article is <b>{label[prediction]}</b>.</p>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<p style="background-color: rgb(254, 242, 242); 
                color: rgb(153, 27, 27);
                font-size: 20px;
                border-radius: 7px;
                padding-left: 12px;
                padding-top: 15px;
                padding-bottom: 15px;
                line-height: 25px;
                text-align: center;">This article is <b>{label[prediction]}</b>.</p>""", unsafe_allow_html=True)
                
            
            with st.expander("Related Articles"):
                for i in sorted[:5]:
                    # st.write(f"""""",unsafe_allow_html=True)
                    st.markdown(f"""
                    <small style="text-align: left;">{data["url"][i].split("/")[2]}</small><br>
                    <a href={data["url"][i]} style="text-align: left;">{data["title"][i]}</a>
                    """, unsafe_allow_html=True)
    else:
        st.markdown(f"""<p style="background-color: rgb(254, 242, 242); 
        color: rgb(153, 27, 27);
        font-size: 20px;
        border-radius: 7px;
        padding-left: 12px;
        padding-top: 15px;
        padding-bottom: 15px;
        line-height: 25px;
        text-align: center;">Can't scrap article from this link.</p>""", unsafe_allow_html=True)
