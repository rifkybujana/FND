import numpy as np
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

"""
this script contain every language preprocessing class

Author: Rifky Bujana Bisri
Email : rifkybujanabisri@gmail.com
"""

def GetObject(lang):
    try:
        pClass = globals()[lang]
        return pClass()
    except:
        print("Wrong language parameter")
        return None

class English:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def GetWordnetPos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        return tag_dict.get(tag, wordnet.NOUN)

    def Preprocess(self, text):
        text = text.lower()
        text = text.split()
        text = ' '.join([re.sub(r'[^A-Za-z0-9]', '', w) for w in text]).split()
        text = ' '.join([self.lemmatizer.lemmatize(w, Get_Wordnet_Pos(w)) for w in text])
        text = re.sub(r'\d+', '[NUM]', x[i])

        return text

class Bahasa:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def Preprocess(self, text):
        text = self.stemmer.stem(text)
        text = re.sub(r'\d+', '[NUM]', text)

        return text