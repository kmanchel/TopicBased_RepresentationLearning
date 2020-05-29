import sys
import os.path
import numpy as np
import pandas as pd
import argparse
import gensim
import gensim.corpora as corpora
from gensim.models import nmf
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
from operator import itemgetter
import ast
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1, help='Dataset for task, 1 for irony, 2 for nepal disaster, 3 for bias')
parser.add_argument('--topic', type=int, default=1, help='Topic modeling option, 1 for lda, 2 for nmf')
parser.add_argument('--numtopics', type=int, default=5, help='Number of topics')
parser.add_argument('--mincount', type=int, default=5, help='Min_count for w2v')
parser.add_argument('--worddim', type=int, default=300, help='Word embedding dimension')
args = parser.parse_args()

# initialize variables
file_name = 'irony.csv'
if args.dataset == 2:
    file_name = 'preprocessed_nepal.csv'
if args.dataset == 3:
	file_name = 'bias_only_3k.csv'
num_topics = args.numtopics
min_count = args.mincount
word_dim = args.worddim
substr = 'irony'
if (args.dataset == 2):
    substr = 'nepal'
if (args.dataset == 3):
	substr = 'partisan'

# read dataset
df = pd.read_csv(file_name)
df_text = df['text']

texts = []
for sentence in df_text:
    words = sentence.split(' ')
    texts.append(words)

# create dictionary
print('--- creating dictionary ---')
id2word = corpora.Dictionary(texts)
id2word.save('./{}_tmp/dfid2word'.format(substr))

# create corpus
print('--- creating corpus for topic modeling ---')
corpus = [id2word.doc2bow(text) for text in texts]

# topic model
print('--- training topic model ---')
topic_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, passes=1)
if (args.topic == 2):
    topic_model = nmf.Nmf(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, passes=1)
topic_model.save('./{}_tmp/topic_model'.format(substr))

topic_model = gensim.models.LdaModel.load('./{}_tmp/topic_model'.format(substr))
id2word = gensim.corpora.Dictionary.load('./{}_tmp/dfid2word'.format(substr))

# split words based on topics
print('--- creating topic and all corpus ---')
for texts in df['text']:
    words = texts.split(' ')
    bow = id2word.doc2bow(words)
    topic_probs = topic_model[bow]
    topic = max(topic_probs, key=itemgetter(1))[0]
    with open('./{}_tmp/datacorpus_'.format(substr) + str(topic) + '.txt', 'a') as f:
        f.write(' '.join(words) + '\n')

with open('./{}_tmp/datacorpus_all.txt'.format(substr), 'a') as f:
    for texts in df['text']:
        words = texts.split(' ')
        f.write(' '.join(words) + '\n')

# training w2v
print('--- training w2v ---')
for i in range(num_topics):
    if not os.path.isfile('./{}_tmp/datacorpus_'.format(substr) + str(i) + '.txt'):
        continue

    model = Word2Vec(LineSentence('./{}_tmp/datacorpus_'.format(substr) + str(i) + '.txt'), size=word_dim, window=5, min_count=min_count, workers=multiprocessing.cpu_count())
    model.save('./{}_tmp/datacorpus_word2vec_'.format(substr) + str(i) + '.model')
    model.wv.save('./{}_tmp/datacorpus_word2vec_'.format(substr) + str(i) + '_wv.model')

model = Word2Vec(LineSentence('./{}_tmp/datacorpus_all.txt'.format(substr)), size=word_dim, window=5, min_count=min_count, workers=multiprocessing.cpu_count())
model.save('./{}_tmp/datacorpus_word2vec_all.model'.format(substr))
model.wv.save('./{}_tmp/datacorpus_word2vec_all_wv.model'.format(substr))

# create embeddings from topic-independent w2v
print('--- creating topic-independent embeddings ---')
np.random.seed(43)
w2v_embeddings = np.zeros((len(df), word_dim))
w2v_rand = np.random.uniform(-0.8, 0.8, word_dim)

# Compute word2vec embeddings.
w2v_model = gensim.models.KeyedVectors.load('./{}_tmp/datacorpus_word2vec_all.model'.format(substr))

# Find all tweets with topic i.
for i in range(len(df)):  
    # Compute average word2vec embeddings.
    average_w2v_embeddings = np.zeros((1, word_dim))

    text = df.loc[i, 'text'].split()
    for word in text:
        if word in w2v_model:
            average_w2v_embeddings[0] += w2v_model[word]
        else:
            average_w2v_embeddings += w2v_rand
    
    average_w2v_embeddings /= len(text)
    w2v_embeddings[i] = average_w2v_embeddings

df['embeddings'] = w2v_embeddings.tolist()
df.to_csv('./{}_tmp/datacorpus_embed_all.csv'.format(substr), index=False)

# create embeddings from topic-dependent w2v's
print('--- creating topic-dependent embeddings ---')
topics = []
w2v_embeddings = np.zeros((len(df), word_dim))

# Compute topic distribution for each tweet.
for i in range(len(df)):
    text = df.loc[i, 'text'].split()
    text = id2word.doc2bow(text)

    topic_probs = topic_model[text]
    if (len(topic_probs) == 0):
        # If text is too short, assign random topic.
        topic = np.random.choice(int(num_topics))

        while (not os.path.isfile('./{}_tmp/datacorpus_word2vec_'.format(substr) + str(topic) + '_wv.model')):
            topic = np.random.choice(int(num_topics))
    else:
        # Compute most prevalant topic.
        topic = max(topic_probs, key=itemgetter(1))[0]

    topics.append(topic)

# Compute topic-dependent embeddings for each tweet.
for i in range(num_topics):
    path = './{}_tmp/datacorpus_word2vec_'.format(substr) + str(i) + '_wv.model'
    if (not os.path.isfile(path)):
        continue

    w2v_model = gensim.models.KeyedVectors.load(path)

    # Find all tweets with topic i.
    for j, topic in enumerate(topics):
        if topic == i:       
            # Compute average word2vec embeddings.
            average_w2v_embeddings = np.zeros((1, word_dim))

            text = df.loc[j, 'text'].split()
            for word in text:
                if word in w2v_model:
                    average_w2v_embeddings[0] += w2v_model[word]
                else:
                    average_w2v_embeddings += w2v_rand
            
            average_w2v_embeddings /= len(text)
            w2v_embeddings[j] = average_w2v_embeddings

df['embeddings'] = w2v_embeddings.tolist()
df.to_csv('./{}_tmp/datacorpus_embed_topics.csv'.format(substr), index=False)

# classification
print('--- classification ---')
all_embed_path = './{}_tmp/datacorpus_embed_all.csv'.format(substr)
topic_embed_path = './{}_tmp/datacorpus_embed_topics.csv'.format(substr)

# Read input files.
df_all_embed = pd.read_csv(all_embed_path)
df_topic_embed = pd.read_csv(topic_embed_path)

# Create train/test sets for all embeddings file and extract embedding features.
test_all_embed = df_all_embed.sample(frac=0.2, random_state=23)
train_all_embed = df_all_embed.drop(test_all_embed.index)

test_all_embed_features = np.asarray([ast.literal_eval(x) for x in test_all_embed['embeddings']])
train_all_embed_features = np.asarray([ast.literal_eval(x) for x in train_all_embed['embeddings']])

# Create train/test sets for topic embeddings file and extract embedding features.
test_topic_embed = df_topic_embed.sample(frac=0.2, random_state=23)
train_topic_embed = df_topic_embed.drop(test_topic_embed.index)

test_topic_embed_features = np.asarray([ast.literal_eval(x) for x in test_topic_embed['embeddings']])
train_topic_embed_features = np.asarray([ast.literal_eval(x) for x in train_topic_embed['embeddings']])

'''
# Train and evaluate accuracy of GBC classifier with topic-independent embedding features. 
if (args.dataset == 1 or args.dataset == 3):
    gbc_model = GBC(random_state=1).fit(train_all_embed_features, train_all_embed['label'])
    y_pred = gbc_model.predict(test_all_embed_features)
    f1 = f1_score(test_all_embed['label'], y_pred, average='micro')
    print('Topic-independent GBC micro F1 score: %.4f' % (f1))
else:
    gbc_model = GBC(random_state=1).fit(train_all_embed_features, train_all_embed['relevance_label'])
    y_pred = gbc_model.predict(test_all_embed_features)
    f1 = f1_score(test_all_embed['relevance_label'], y_pred, average='micro')
    print('Topic-independent GBC micro F1 score: %.4f' % (f1)) 

# Train and evaluate accuracy of GBC classifier with topic-dependent embedding features. 
if (args.dataset == 1 or args.dataset == 3):
    gbc_model = GBC(random_state=1).fit(train_topic_embed_features, train_topic_embed['label'])
    y_pred = gbc_model.predict(test_topic_embed_features)
    f1 = f1_score(test_all_embed['label'], y_pred, average='micro')
    print('Topic-dependent GBC micro F1 score: %.4f' % (f1))
else:
    gbc_model = GBC(random_state=1).fit(train_topic_embed_features, train_topic_embed['relevance_label'])
    y_pred = gbc_model.predict(test_topic_embed_features)
    f1 = f1_score(test_all_embed['relevance_label'], y_pred, average='micro')
    print('Topic-dependent GBC micro F1 score: %.4f' % (f1))    
'''
# Train and evaluate accuracy of LR classifier with topic-independent embedding features. 
if (args.dataset == 1 or args.dataset == 3):
    lr_model = LR(penalty='l2').fit(train_all_embed_features, train_all_embed['label'])
    test_accuracy = lr_model.score(test_all_embed_features, test_all_embed['label'])
    print('Topic-independent LR Test Accuracy: %.4f' % (test_accuracy * 100))
else:
    lr_model = LR(penalty='l2').fit(train_all_embed_features, train_all_embed['relevance_label'])
    test_accuracy = lr_model.score(test_all_embed_features, test_all_embed['relevance_label'])
    print('Topic-independent LR Test Accuracy: %.4f' % (test_accuracy * 100)) 

# Train and evaluate accuracy of LR classifier with topic-dependent embedding features. 
if (args.dataset == 1 or args.dataset == 3):
    lr_model = LR(penalty='l2').fit(train_topic_embed_features, train_topic_embed['label'])
    test_accuracy = lr_model.score(test_topic_embed_features, test_topic_embed['label'])
    print('Topic-dependent LR Test Accuracy: %.4f' % (test_accuracy * 100))
else:
    lr_model = LR(penalty='l2').fit(train_topic_embed_features, train_topic_embed['relevance_label'])
    test_accuracy = lr_model.score(test_topic_embed_features, test_topic_embed['relevance_label'])
    print('Topic-dependent LR Test Accuracy: %.4f' % (test_accuracy * 100))    


