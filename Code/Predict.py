import tensorflow as tf
import numpy as np
import argparse
import sys

from Scraper import Scrap
from Preprocess import *

"""
This script are used to predict an article from a given link

Author: Rifky Bujana Bisri
email : rifkybujanabisri@gmail.com
"""

def LoadModel(path):
    """
    Load the model from a path

    ### Parameter\n
    path : path to the saved model folder (dtype: `string`)

    ### Result\n
    return the model object
    """

    return tf.keras.models.load_model(path)

def Predict(model, preprocess, text):
    """
    Make a prediction of the article from the url

    ### Parameter\n
    model : model object that will be used to make the prediction (dtype: `object`)\n
    preprocess : preprocess object, `check Preprocess.py`\n
    text : article text (dtype: `string`)

    ### Result\n
    return the model prediction from the article which represented by float from 0 - 1 (dtype: `float`)\n
    0 - 0.5 mean that the model predict its valid
    0.5 - 1 mean that the model predict its fake
    """

    text = preprocess.Preprocess(text)

    return model.predict(np.array([text]))[0][0]

relative_path = """./Data/Model/"""

if __name__ == "__main__":

    ############################################# ARGUMENTS ################################################

    parser = argparse.ArgumentParser(description="This tools is used to predict a news from a given url is true or false")
    parser.add_argument('url', type=str, help='url of the article you want to predict')
    parser.add_argument('lang', type=str, help="""article language [English, Bahasa], default: Bahasa""", default='Bahasa')
    args = parser.parse_args()

    ########################################### END ARGUMENTS ##############################################

    text = Scrap(args.url, args.lang)

    if not text:
        sys.exit()

    model = LoadModel(relative_path + args.lang.lower())
    preprocess = GetObject(args.lang)

    prediction = Predict(model, preprocess, text)
    print(prediction)