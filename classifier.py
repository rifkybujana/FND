from keras.preprocessing.text import Tokenizer as tk
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import load_model

import pickle
import numpy as np

from LogisticRegression import LogisticRegression
from Preprocess import Tokenizer, Encoder

def Classify(text, language, model):
    tokenizer = Tokenizer(language)
    data = tokenizer.Tokenize(text)

    if model == 'Logistic Regression':    
        model   = pickle.load(open('Data/Model/Logistic Regression/' + language + '.pkl', 'rb'))
        encoder = pickle.load(open('Data/Encoder/' + language + '.pkl', 'rb'))

        data = encoder.OneHot(data)
        data = np.array(data).reshape(1, len(data))

        prediction = model.predict_prob(data)

    else:    
        model   = load_model('Data/Model/CRNN/' + language + '.h5')
        encoder = pickle.load(open('Data/Tokenizer/' + language + '.pkl', 'rb'))

        data = ' '.join(data)
        data = encoder.texts_to_sequences(np.array([data, ]))
        data = pad_sequences(data, maxlen=300)

        prediction = model.predict(data)[0]

    return prediction