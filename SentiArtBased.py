#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:00:00 2020
Sentiment analysis tool for literary texts based on the SentiArt (Jacobs AM 2019, doi:10.3389/frobt.2019.00053) model.
@author: wiskom93@zedat.fu-berlin and terlaul96@zedat.fu-berlin.de
"""
# get packages
import codecs
import pandas as pd
import numpy as np
import glob
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag_sents, download
import matplotlib.pyplot as plt
import seaborn as sns


""" get the table with sentiment values (e.g., AAPz, fear_z).
these are based on a vector space model (w2v, skipgram, 300d) and the label list published in:
https://www.frontiersin.org/articles/10.3389/fnhum.2017.00622/full#supplementary-material; https://www.frontiersin.org/articles/10.3389/fpsyg.2020.574746/full
the values for each word are: AAPz,ang_z,fear_z,disg_z,hap_z,sad_z,surp_z
they provide the affective-aesthetic potential (AAP) and discrete emotion values (anger, fear, disgust, sadness and surprise), all standardized (z-values),
for each word, based on their semantic relatedness (as computed by w2v) with labels (semantic anchors) described in the publications mentioned in readme.md
"""
song_order = ['dylan_low_lit/going_going_gone.txt',
              'dylan_low_lit/lay_lady_lay.txt',
              'dylan_low_lit/i_threw_it_all_away',
              'dylan_low_lit/abandoned_love']


def read_tokenize(file):
    with codecs.open(file, 'r', 'utf-8') as f:
        raw = f.read().replace('\n', '. ').replace('..', '.').replace('!', ' ')
    sents = sent_tokenize(raw)
    tokens = [[t for t in word_tokenize(s) if t.isalpha()] for s in sents]
    return sents, tokens


def pos_tag(tokens):
    return [pos_tag(t) for t in tokens]


TC = '250kSentiArt_EN.csv' # for English texts
sa = pd.read_csv(TC)  # csv is way faster
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


def plot_only_aaps(df, title):
    print(df.head())
    df.set_index(df.index,inplace=True)
    ax = df.plot(kind='bar',alpha=0.75, rot=0, legend=False)
    x_offset = -0.4
    y_offset = 0.02
    for p in ax.patches:
        if p.get_height() < 0:
            ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2. + x_offset, y_offset))
        else:
            ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2. + x_offset, p.get_height() + y_offset))
    plt.title(title)
    plt.xlabel("Sentence #")
    plt.ylabel("AAPs")
    plt.show()


def calc_aap():
    file = ""
    sa_lines = pd.DataFrame()
    for file in song_order:
        _, tokens = read_tokenize(file)
        #tokens = pos_tag_sents(tokens, tagset="universal")  # simplified pos tagging
        sentiment_values = get_sentiments(sa, tokens)
        sentiment_values["tokens"] = tokens
        sa_lines = sa_lines.append(sentiment_values[["tokens", "aap"]], ignore_index=True)
        #sentiment_values = round(sentiment_values, 3)
        #plot(sentiment_values, file[:-4])
        #plot_only_aaps(sentiment_values.loc[:,['aap', 'tokens']], file[:-4])

    sa_lines = sa_lines[sa_lines["tokens"].str.len() > 0]  # remove empty lines from initial text parsing
    sa_lines = sa_lines.reset_index(drop=True)
    return sa_lines

