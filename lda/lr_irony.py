from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
import numpy as np
import ast
import os.path
import logging
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

all_embed_path = 'datacorpus_embed_all.csv'
topic_embed_path = 'datacorpus_embed_topics.csv'

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

# Train and evaluate accuracy of LR classifier with topic-independent embedding features. 
lr_model = LR(penalty='l2').fit(train_all_embed_features, train_all_embed['label'])
test_accuracy = lr_model.score(test_all_embed_features, test_all_embed['label'])
print('Topic-independent LR Test Accuracy: %.4f' % (test_accuracy * 100))

# Train and evaluate accuracy of LR classifier with topic-dependent embedding features. 
lr_model = LR(penalty='l2').fit(train_topic_embed_features, train_topic_embed['label'])
test_accuracy = lr_model.score(test_topic_embed_features, test_topic_embed['label'])
print('Topic-dependent LR Test Accuracy: %.4f' % (test_accuracy * 100))
