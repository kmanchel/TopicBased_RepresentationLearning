import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora

df = pd.read_csv('trainA.csv')
df_text = df['text']

texts = []
for sentence in df_text:
    words = sentence.split(' ')
    texts.append(words)

# create cictionary
id2word = corpora.Dictionary(texts)
id2word.save('dfid2word')

# create corpus
corpus = [id2word.doc2bow(text) for text in texts]

# lda model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, 
        random_state=100, update_every=1, passes=1)
lda_model.save('lda_model')