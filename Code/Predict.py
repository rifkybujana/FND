import tensorflow as tf
import numpy as np
import re
import argparse
import sys

from Scraper import Scrap

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

"""
This script are used to predict an article from a given link

Author: Rifky Bujana Bisri
email : rifkybujanabisri@gmail.com
"""

def CreateStemmer():
    """
    Create indonesian stemmer

    ### Result\n
    indonesian "Sastrawi" stemmer object (dtype: `object`)
    """

    factory = StemmerFactory()
    return factory.create_stemmer()

def LoadModel(path):
    """
    Load the model from a path

    ### Parameter\n
    path : path to the saved model folder (dtype: `string`)

    ### Result\n
    return the model object
    """

    return tf.keras.models.load_model(path)

def Predict(model, stemmer, text):
    """
    Make a prediction of the article from the url

    ### Parameter\n
    model : model object that will be used to make the prediction (dtype: `object`)\n
    text : article text (dtype: `string`)

    ### Result\n
    return the model prediction from the article which represented by float from 0 - 1 (dtype: `float`)\n
    0 - 0.5 mean that the model predict its valid
    0.5 - 1 mean that the model predict its fake
    """

    text = stemmer.stem(text)
    text = re.sub(r'\d+', '[NUM]', text)

    return model.predict(np.array([text]))[0][0]

if __name__ == "__main__":

    ############################################# ARGUMENTS ################################################

    parser = argparse.ArgumentParser(description="This tools is used to predict a news from a given url is true or false")
    parser.add_argument('url', type=str, help='url of the article you want to predict')
    parser.add_argument('--model_path', type=str, help='your own model, default: .\Data\Model\indonesian', default='.\Data\Model\indonesian')
    args = parser.parse_args()

    ########################################### END ARGUMENTS ##############################################

    text = Scrap(args.url)

    if not text:
        sys.exit()

    stemmer = CreateStemmer()
    model = LoadModel(args.model_path)
    prediction = Predict(model, stemmer, text)

    if prediction >= 0.5:
        print("Fake")
    else:
        print("Valid")