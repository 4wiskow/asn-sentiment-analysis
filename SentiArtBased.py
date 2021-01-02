#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:00:00 2020
Sentiment analysis tool for literary texts based on the SentiArt (Jacobs AM 2019, doi:10.3389/frobt.2019.00053) model.
@author: wiskom93@zedat.fu-berlin
"""
# get packages
import codecs
import pandas as pd
import glob
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag_sents, download
import matplotlib.pyplot as plt


""" get the table with sentiment values (e.g., AAPz, fear_z).
these are based on a vector space model (w2v, skipgram, 300d) and the label list published in:
https://www.frontiersin.org/articles/10.3389/fnhum.2017.00622/full#supplementary-material; https://www.frontiersin.org/articles/10.3389/fpsyg.2020.574746/full
the values for each word are: AAPz,ang_z,fear_z,disg_z,hap_z,sad_z,surp_z
they provide the affective-aesthetic potential (AAP) and discrete emotion values (anger, fear, disgust, sadness and surprise), all standardized (z-values),
for each word, based on their semantic relatedness (as computed by w2v) with labels (semantic anchors) described in the publications mentioned in readme.md
"""


def read_tokenize(file):
    with codecs.open(file, 'r', 'utf-8') as f:
        raw = f.read().replace('\n', '. ').replace('..', '.').replace('!', ' ')
    sents = sent_tokenize(raw)
    tokens = [[t for t in word_tokenize(s) if t.isalpha()] for s in sents]
    return sents, tokens


def pos_tag(tokens):
    return [pos_tag(t) for t in tokens]


TC = '250kSentiArt_EN.csv' # for English texts
# TC = '120kSentiArt_DE.xlsx' # for German texts
#sa = pd.read_excel(TC)
sa = pd.read_csv(TC)  # csv is way faster
print(sa.head())
#sa = sa.drop(columns="Unnamed: 0")  # drop index column
# download('universal_tagset')  # before pos tagging, need to download tag set


# compute mean AAP (or mean fear etc.) per sentence
def plain_mean(sa, t):
    dt = sa.query('word in @t')
    vals = [dt.AAPz.mean(),
                     dt.fear_z.mean(),
                     dt.ang_z.mean(),
                     dt.disg_z.mean(),
                     dt.hap_z.mean(),
                     dt.sad_z.mean(),
                     dt.surp_z.mean()]

    return vals


# word-frequency weighted values
def freq_weighted(sa, t):
    dt = sa.query('word in @t')
    dt = dt.assign(freq=lambda x: [t.count(w) / len(x.word) for w in x.word])  # word freq in t
    vals = [sum(dt.AAPz * dt.freq),
             sum(dt.fear_z * dt.freq),
             sum(dt.ang_z * dt.freq),
             sum(dt.disg_z * dt.freq),
             sum(dt.hap_z * dt.freq),
             sum(dt.sad_z * dt.freq),
             sum(dt.surp_z * dt.freq)]

    return vals


# compute sentiment values for each sentence
def get_sentiments(sa, tokens_by_sents):
    vals = pd.DataFrame(columns=["aap", "ang", "fear", "disg", "hap", "sad", "surp"])
    for idx, t in enumerate(tokens_by_sents):
        vals.loc[idx] = freq_weighted(sa, t)

    return vals


# panda & save results
def to_df(tokens, sent_mean_AAPz, sent_mean_fear_z):
    df = pd.DataFrame()
    df['sent'] = tokens
    df['AAPz'] = sent_mean_AAPz
    df['fear_z'] = sent_mean_fear_z
    df = round(df,3)
    # df.to_csv('results.txt')
    return df


# plot AAPz, fear_z etc.
def plot(df, title):
    df.set_index(df.index,inplace=True)
    df.plot(kind='bar',alpha=0.75, rot=0)
    plt.title(title)
    plt.xlabel("Sentence #")
    plt.ylabel("Sentiment Value (z)")
    plt.show()


file = ""
for file in glob.glob("dylan_low_lit/*"):
    sents, tokens = read_tokenize(file)
    #tokens = pos_tag_sents(tokens, tagset="universal")  # simplified pos tagging
    sentiment_values = get_sentiments(sa, tokens)
    sentiment_values["tokens"] = tokens
    sentiment_values = round(sentiment_values, 3)
    plot(sentiment_values, file[:-4])

    print(file)
    print(sentiment_values["aap"].mean())
