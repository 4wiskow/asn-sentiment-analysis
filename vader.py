#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:24:58 2020
Vader basic sentiment analysis tool for literary texts
@author: terlaul96@zedat.fu-berlin.de
"""
# get packages/requirements
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import codecs
import glob
from nltk import sent_tokenize, word_tokenize

vader = SentimentIntensityAnalyzer()

def read_data(fname: str, colnames=['text']):
    "Read in test data into a Pandas DataFrame"
    df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
    #print(df)
    return df


def scores(df):
    # Create Prediction column based on Polarity Score
    # converts negative compound scores to -1 and positive compound scores to 1
    #df['Prediction'] = df['text'].apply(lambda x: 1 if vader.polarity_scores(x)['compound'] >= 0 else -1)
    df['Prediction'] = df['text'].apply(lambda x: vader.polarity_scores(x)['compound'])
    df['pos'] = df['text'].apply(lambda x: vader.polarity_scores(x)['pos'])
    df['neg'] = df['text'].apply(lambda x: vader.polarity_scores(x)['neg'])
    df['neutral'] = df['text'].apply(lambda x: vader.polarity_scores(x)['neu'])
    #df['score'] = vader.polarity_scores(df['text'])['compound']
    print(df.head())
    return df

def plot(df, title):
    df.set_index(df.index,inplace=True)
    df.plot(kind='bar',alpha=0.75, rot=0)
    plt.title(title)
    plt.xlabel("Sentence #")
    plt.ylabel("Vader prediction")
    plt.show()

file = ""
for file in glob.glob("dylan_low_lit/*"):
    print(file)
    df = read_data(file)
    prediction_values = scores(df)
    plot(prediction_values, file[:-4])
