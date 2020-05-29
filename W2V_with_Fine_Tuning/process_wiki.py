import gensim
import multiprocessing
import logging
import os.path
import sys
import numpy as np
from operator import itemgetter
import pandas as pd

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

lda_model = gensim.models.LdaModel.load('data/lda_model')
id2word = gensim.corpora.Dictionary.load('data/dfid2word')

df = pd.read_csv('data/bias_only_3k.csv')

for i in range(40):
    if os.path.exists('data/datacorpus_' + str(i) + '.txt'):
        os.remove('data/datacorpus_' + str(i) + '.txt')

for texts in df['text']:
    words = texts.split()
    bow = id2word.doc2bow(words)
    topic_probs = lda_model[bow]
    topic = max(topic_probs, key=itemgetter(1))[0]
    with open('data/datacorpus_' + str(topic) + '.txt', 'a') as f:
        f.write(' '.join(words) + '\n')

with open('data/datacorpus_all.txt', 'w') as f:
    for texts in df['text']:
        words = texts.split(' ')
        f.write(' '.join(words) + '\n')
