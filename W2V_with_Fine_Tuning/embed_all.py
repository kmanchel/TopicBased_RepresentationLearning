import gensim
import os.path
import logging
import pandas as pd
import numpy as np
import sys
from operator import itemgetter
import os

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

df = pd.read_csv('data/bias_only_3k.csv')

np.random.seed(43)
w2v_embeddings = np.zeros((len(df), 300))
w2v_rand = np.random.uniform(-0.8, 0.8, 300)
w2v_model = gensim.models.KeyedVectors.load('data/datacorpus_word2vec_all.model')

# Find all tweets with topic i.
for i in range(len(df)):  
    # Compute average word2vec embeddings.
    average_w2v_embeddings = np.zeros((1, 300))

    text = df.loc[i, 'text'].split()
    for k, word in enumerate(text):
        if word in w2v_model:
            average_w2v_embeddings += w2v_model[word]
        else:
            average_w2v_embeddings += w2v_rand
    
    average_w2v_embeddings /= len(text)
    w2v_embeddings[i] = average_w2v_embeddings

df['embeddings'] = w2v_embeddings.tolist()
df.to_csv('data/datacorpus_embed_all_bias.csv', index=False)
