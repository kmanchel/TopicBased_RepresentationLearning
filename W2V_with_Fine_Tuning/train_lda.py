import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
import logging
import os.path
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

df = pd.read_csv('data/bias_only_3k.csv')
df_text = df['text']

texts = []
for sentence in df_text:
    words = sentence.split()
    texts.append(words)

# create cictionary
id2word = corpora.Dictionary(texts)
id2word.save('data/dfid2word')

# create corpus
corpus = [id2word.doc2bow(text) for text in texts]

# lda model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=10, 
    random_state=100, update_every=1, passes=10)
lda_model.save('data/lda_model')
