import re
import nltk
import numpy as np

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from joblib import Parallel, delayed

class Tokenizer():
    def __init__(self, language):
        self.language = language
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def Tokenize(self, text):
        text = text.lower()
        text = text.split()
        text = ' '.join(Parallel(n_jobs=-1)(delayed(re.sub)(r'[^A-Za-z0-9]+', '', w) for w in text))

        if self.language == 'english':
            text = text.split()
            text = Parallel(n_jobs=-1)(delayed(self.lemmatizer.lemmatize)(i) for i in text)
        else:
            text = self.stemmer.stem(text).split()

        text = ' '.join(Parallel(n_jobs=-1)(delayed(re.sub)(r'\d+', ' <num> ', w) for w in text)).split()
        text = list(dict.fromkeys(text))
        text = [w for w in text if not w in self.stopwords and len(w) > 2]

        return text

class Encoder():
    def __init__(self, vocab = None):
        if vocab:
            self.vocab = vocab

    def CreateVocab(self, data, language):
        vocab = ['unk', 'num']
        for i in data:
            vocab += i

        tokenizer = Tokenizer(language)
        vocab = tokenizer.Tokenize(' '.join(vocab))
        self.vocab = vocab

        return vocab

    def OneHot(self, text):
        result = [0] * len(self.vocab)
        for i in text:
            if i == "<num>":
                result[self.vocab.index("num")] = 1
            else:
                try:
                    result[self.vocab.index(i)] = 1
                except:
                    result[self.vocab.index("unk")] = 1
        
        return result