from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
import os.path
import sys

for i in range(20):
    if not os.path.isfile('datacorpus_' + str(i) + '.txt'):
        continue

    model = Word2Vec(LineSentence('datacorpus_' + str(i) + '.txt'), size=300, window=5, min_count=5,
        workers=multiprocessing.cpu_count())
    model.save('datacorpus_word2vec_' + str(i) + '.model')
    model.wv.save('datacorpus_word2vec_' + str(i) + '_wv.model')

model = Word2Vec(LineSentence('datacorpus_all.txt'), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save('datacorpus_word2vec_all.model')
model.wv.save('datacorpus_word2vec_all_wv.model')
