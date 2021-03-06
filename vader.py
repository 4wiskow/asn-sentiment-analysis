#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vader basic sentiment analysis tool for literary texts
@author: terlaul96@zedat.fu-berlin.de and wiskom93@zedat.fu-berlin
"""
# get packages/requirements
import SentiArtBased
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import codecs
import glob
from nltk import sent_tokenize, word_tokenize

# Initialize the VADER sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()



def read_data(fname: str, colnames=['text']):
    """Read in test data into a Pandas DataFrame"""
    df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
    #print("TEST",df)
    return df


def scores(df):
    """Score text based on VADER"""
    # Create Prediction column based on Polarity Score
    # converts negative compound scores to -1 and positive compound scores to 1
    df['vader_compound'] = df['text'].apply(lambda x: vader.polarity_scores(x)['compound'])
    df['vader_label'] = (df["vader_compound"] >= 0).astype("int32")
    df['vader_pos'] = df['text'].apply(lambda x: vader.polarity_scores(x)['pos'])
    df['vader_neg'] = df['text'].apply(lambda x: vader.polarity_scores(x)['neg'])
    #df['neutral'] = df['text'].apply(lambda x: vader.polarity_scores(x)['neu'])
    #df['score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    print(df.head())
    return df


def plot(df, title):
    """Plot scores for song"""
    df.set_index(df.index,inplace=True)
    df.plot(kind='bar',alpha=0.75, rot=0)
    plt.title(title)
    plt.xlabel("Sentence #")
    plt.ylabel("Vader prediction")
    plt.show()


def get_vader_lexicon():
    """Get lexicon of tokens known to VADER"""
    #return analyzer.make_lex_dict()
    return SentimentIntensityAnalyzer().make_lex_dict()


def calc_vader_scores():
    """Calculate scores and hit rate based on VADER"""
    vader_scores = pd.DataFrame()
    for file in SentiArtBased.SONGS_ORDERED:
        print("File", file)
        df = read_data(file)
        prediction_values = scores(df)
        vader_scores = vader_scores.append(prediction_values)
        #plot(prediction_values, file[:-4])

    vader_scores = vader_scores.reset_index(drop=True)
    tokens = SentiArtBased.get_tokens_per_line(SentiArtBased.SONGS_ORDERED)
    hit_rate = SentiArtBased.per_line_hit_rate(tokens, get_vader_lexicon().keys())
    vader_scores["hit_rate"] = hit_rate
    #print("Hit rate Vader:", hit_rate)
    return vader_scores
