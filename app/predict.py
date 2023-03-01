import pandas as pd
import os
import pickle
import numpy as np
import xgboost
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

cv = CountVectorizer(lowercase=False)


def load():
    os.chdir("..")
    df = pd.read_csv('Data/tablesBalanced.csv', sep=';', encoding='windows-1251', header=None)
    df[0] = df[0].str.lower()
    X = df[1]
    Y = df[0]
    cv.fit(X)

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    le_name_mapping = dict(zip(Y_encoded, Y))

    loaded_model = pickle.load(open('Data/xgbModel.sav', 'rb'))
    return le_name_mapping, loaded_model


def proba(input):
    testData = [input]
    testData = cv.transform(testData)
    probabilities = loaded_model.predict_proba(testData)
    arr = np.empty((0, 2))
    for n in range(len(probabilities[0])):
        arr = np.append(arr, np.array([[le_name_mapping[n], probabilities[0][n]]]), axis=0)

    arr = arr[arr[:, 1].argsort()[::-1]]
    return arr


le_name_mapping, loaded_model = load()

