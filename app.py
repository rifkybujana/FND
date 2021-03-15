import streamlit as st
import requests
import classifier

from bs4 import BeautifulSoup


st.write("""# Fake News Detections AI""")

text    = st.text_area("Enter the full article or the link")
submit  = st.button("Enter")


st.sidebar.write("""## Settings""")

language    = st.sidebar.selectbox('language', ['english', 'indonesian'])
model_type  = st.sidebar.selectbox('Machine Learning Model', ['Logistic Regression', 'CRNN'])



if submit:
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
                except:
                    st.write("Cant Process the article in this link")
            else:
                data = text

    if data:
        prediction = classifier.Classify(data, language, model_type)

        if prediction >= 0.5:
            st.write("""### i {}% sure its fake""".format(str((prediction - 0.5) * 200)))
        else:
            st.write("""### i {}% sure its valid""".format(str((0.5 - prediction) * 200)))