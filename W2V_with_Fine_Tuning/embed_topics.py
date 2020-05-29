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
lda_model = gensim.models.LdaModel.load('data/lda_model')
id2word = gensim.corpora.Dictionary.load('data/dfid2word')
word2vec_path = 'data/datacorpus_word2vec'

np.random.seed(37)
w2v_rand = np.random.uniform(-0.8, 0.8, 300)
w2v_embeddings_opt1 = np.zeros((len(df), 300))
w2v_embeddings_opt2 = np.zeros((len(df), 300))

num_topics = 10
topics = []

# Compute topic distribution for each tweet.
for i in range(len(df)):
    text = df.loc[i, 'text'].split()
    text = id2word.doc2bow(text)

    topic_probs = lda_model[text]
    if (len(topic_probs) == 0):
        # If text is too short, assign random topic.
        topic = np.random.choice(int(num_topics))

        while (not os.path.isfile(word2vec_path + '_' + str(topic) + '_wv_opt1.model')):
            topic = np.random.choice(int(num_topics))
    else:
        # Compute most prevalant topic.
        topic = max(topic_probs, key=itemgetter(1))

        while (not os.path.isfile(word2vec_path + '_' + str(topic[0]) + '_wv_opt1.model')):
            if len(topic) == 2:
                topic = [topic]
                
            topic_probs = list(set(topic_probs) - set(topic))
            topic = max(topic_probs, key=itemgetter(1))

        topic = topic[0]

    topics.append(topic)

# Compute topic-dependent embeddings for each tweet.
for i in range(num_topics):
    path1 = 'data/datacorpus_word2vec_' + str(i) + '_wv_opt1.model'
    path2 = 'data/datacorpus_word2vec_' + str(i) + '_wv_opt2.model'
    if (not os.path.isfile(path1)):
        continue

    w2v_opt1 = gensim.models.KeyedVectors.load(path1)
    w2v_opt2 = gensim.models.KeyedVectors.load(path2)

    # Find all tweets with topic i.
    for j, topic in enumerate(topics):
        if topic == i:       
            # Compute average word2vec embeddings.
            average_w2v_embeddings_opt1 = np.zeros((1, 300))
            average_w2v_embeddings_opt2 = np.zeros((1, 300))

            text = df.loc[j, 'text'].split()
            for k, word in enumerate(text):
                if word in w2v_opt1:
                    average_w2v_embeddings_opt1 += w2v_opt1[word]
                else:
                    average_w2v_embeddings_opt1 += w2v_rand
                
                if word in w2v_opt2:
                    average_w2v_embeddings_opt2 += w2v_opt2[word]
                else:
                    average_w2v_embeddings_opt2 += w2v_rand
            
            average_w2v_embeddings_opt1 /= len(text)
            average_w2v_embeddings_opt2 /= len(text)

            w2v_embeddings_opt1[j] = average_w2v_embeddings_opt1
            w2v_embeddings_opt2[j] = average_w2v_embeddings_opt2

df['embeddings'] = w2v_embeddings_opt1.tolist()
df['topics'] = topics
df.to_csv('data/datacorpus_embed_topics_opt1_bias.csv', index=False)

df['embeddings'] = w2v_embeddings_opt2.tolist()
df['topics'] = topics
df.to_csv('data/datacorpus_embed_topics_opt2_bias.csv', index=False)
