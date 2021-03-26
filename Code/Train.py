import tensorflow as tf
import argparse
import pandas as pd

from Preprocess import *
from sklearn.model_selection import train_test_split

"""
This tools is used to train the model with your own dataset

Author: Rifky Bujana Bisri
email : rifkybujanabisri@gmail.com
"""

def ReadData(path):
    """
    Read dataset for training and testing the model\n
    `Note: you need to use `label` and `text` as your dataset columns`

    ### Parameter\n
    path : Dataset .csv path (dtype: `string`)

    ### Result\n
    return 2 numpy array, x and y\n

    x : dataset text numpy array\n
    y : dataset label numpy array
    """

    df = pd.read_csv(path)

    x = df['text'].values
    y = df['label'].values

    if not x or not y:
        print("please use 'text' and 'label' as the column name!")
        return None

    return x, y

def Preprocess(data, preprocess):
    """
    Preprocess all the text in the data

    ### Parameter\n
    data : text dataset (dtype: `numpy array`)\n
    preprocess : preprocessor object (dtype: `object`)

    ### Result\n
    return numpy array that contain preprocessed text
    """

    for i in range(len(data)):
        data[i] = preprocess.Preprocess(data[i])

    return data

def SplitData(x, y, test_size, random_state):
    """
    Splitting dataset into train dataset and test dataset
    
    ### Parameter\n
    x : text dataset (dtype: `numpy array`)\n
    y : label dataset (dtype: `numpy array`)\n
    test_size : test size based on all dataset (dtype: `float`)\n
    random_state : random state type to randomize the dataset (dtype: `int`)

    ### Result\n
    return 4 numpy array which is:\n
    xTrain : text train dataset (dtype: `numpy array`)\n
    xTest : text test dataset (dtype: `numpy array`)\n
    yTrain : label train dataset (dtype: `numpy array`)\n
    yTest : label test dataset (dtype: `numpy array`)
    """

    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def CreateModel(xTrain, vocab_size):
    """
    Create the Convolutional Bidirectional Recurrent Neural Networks Model (CBRNN)

    ### Parameter\n
    xTrain : text train dataset (dtype: `numpy array`)\n
    vocab_size : maximum vocabulary size for the model (dtype: `int`)

    ### Result\n
    return the model object
    """

    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size,standardize=None)
    encoder.adapt(xTrain)

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    return model

def Fit(model, xTrain, xTest, yTrain, yTest, epochs):
    """
    Train the model

    ### Parameter\n
    model : model object (dtype: `object`)\n
    xTrain : text train dataset (dtype: `numpy array`)\n
    xTest : text test dataset (dtype: `numpy array`)\n
    yTrain : label train dataset (dtype: `numpy array`)\n
    yTest : label test dataset (dtype: `numpy array`)\n
    epochs : number of iteration for the model to train from the training dataset (dtype: `int`)
    """

    model.fit(xTrain, yTrain, epochs=epochs, validation_data=(xTest, yTest), validation_steps=30)

def SaveModel(model, path):
    """
    Save the trained model

    ### Parameter\n
    model : model object (dtype: `object`)\n
    path : path to a location for saving this model (dtype: `string`)
    """

    model.save(path)

if __name__ == "__main__":
    
    ############################################# ARGUMENTS ################################################

    parser = argparse.ArgumentParser(description="""This tools is used to create and train the model with your own dataset""")
    parser.add_argument('path', type=str, help="""your dataset path""")
    parser.add_argument('save_path', type=str, help="""where do you want to save the model""")
    parser.add_argument('lang', type=str, help="""dataset language [English, Bahasa], default: Bahasa""", default='Bahasa')
    parser.add_argument('epochs', type=int, help="""number of iteration for the model to train from the training dataset, default: 10""", default=10)
    parser.add_argument('--test_size', type=float, help="""test dataset size based on total from 0 - 1, default: 0.1""", default=0.1)
    parser.add_argument('--random_state', type=int, help="""random state type for randomize the dataset for train and test, default: None""", default=None)
    parser.add_argument('--vocab_size', type=int, help="""just get top x word from the whole dataset, default: 1000""", default=1000)
    args = parser.parse_args()
    
    ########################################### END ARGUMENTS ##############################################

    # read dataset and split the text and label
    x, y = ReadData(args.path)

    # Preprocess the data
    preprocess = GetObject(args.lang)
    x = Preprocess(x, preprocess)

    # split the dataset into train and test dataset
    xTrain, xTest, yTrain, yTest = SplitData(x, y, args.test_size, args.random_state)

    # create CBRNN model
    model = CreateModel(xTrain, args.vocab_size)

    # train the model
    Fit(model, xTrain, xTest, yTrain, yTest, args.epochs)
    
    # evaluate the model
    evaluation = model.evaluate(xTest, yTest)
    print("Accuracy: {}\nLoss: {}".format(evaluation[1], evaluation[0]))

    # save the model
    SaveModel(model, args.save_path)