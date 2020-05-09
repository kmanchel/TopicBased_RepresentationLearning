import gensim
import multiprocessing
import logging
import os.path
import sys
import numpy as np
from operator import itemgetter
import pandas as pd

lda_model = gensim.models.LdaModel.load('lda_model')
id2word = gensim.corpora.Dictionary.load('dfid2word')

df = pd.read_csv('trainA.csv')

for texts in df['text']:
    words = texts.split(' ')
    bow = id2word.doc2bow(words)
    topic_probs = lda_model[bow]
    topic = max(topic_probs, key=itemgetter(1))[0]
    with open('datacorpus_' + str(topic) + '.txt', 'a') as f:
        f.write(' '.join(words) + '\n')

with open('datacorpus_all.txt', 'w') as f:
    for texts in df['text']:
        words = texts.split(' ')
        f.write(' '.join(words) + '\n')
