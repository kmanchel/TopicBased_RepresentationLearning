from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, f1_score
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

all_embed_path = 'data/datacorpus_embed_all_bias.csv'
topic_embed_path_opt1 = 'data/datacorpus_embed_topics_opt1_bias.csv'
topic_embed_path_opt2 = 'data/datacorpus_embed_topics_opt2_bias.csv'

# Read input files.
df_all_embed = pd.read_csv(all_embed_path)
df_topic_embed_opt1 = pd.read_csv(topic_embed_path_opt1)
df_topic_embed_opt2 = pd.read_csv(topic_embed_path_opt2)

'''
Create train/test sets for topic-independent embeddings file and extract embedding features.
'''
test_all_embed = df_all_embed.sample(frac=0.2, random_state=23)
train_all_embed = df_all_embed.drop(test_all_embed.index)

test_all_embed_features = np.asarray([ast.literal_eval(x) for x in test_all_embed['embeddings']])
train_all_embed_features = np.asarray([ast.literal_eval(x) for x in train_all_embed['embeddings']])

'''
Create train/test sets for topic-dependent (with fine-tuning) embeddings file and extract embedding features.
'''
test_topic_embed_opt1 = df_topic_embed_opt1.sample(frac=0.2, random_state=23)
train_topic_embed_opt1 = df_topic_embed_opt1.drop(test_topic_embed_opt1.index)

test_topic_embed_features_opt1 = np.asarray([ast.literal_eval(x) for x in test_topic_embed_opt1['embeddings']])
train_topic_embed_features_opt1 = np.asarray([ast.literal_eval(x) for x in train_topic_embed_opt1['embeddings']])

'''
Create train/test sets for topic-dependent (without fine-tuning) embeddings file and extract embedding features.
'''
test_topic_embed_opt2 = df_topic_embed_opt2.sample(frac=0.2, random_state=23)
train_topic_embed_opt2 = df_topic_embed_opt2.drop(test_topic_embed_opt2.index)

test_topic_embed_features_opt2 = np.asarray([ast.literal_eval(x) for x in test_topic_embed_opt2['embeddings']])
train_topic_embed_features_opt2 = np.asarray([ast.literal_eval(x) for x in train_topic_embed_opt2['embeddings']])

'''
Evaluation
'''
# Train and evaluate accuracy of LR classifier with topic-independent embedding features. 
lr_model = LR(penalty='l2', max_iter=1000, solver='lbfgs').fit(train_all_embed_features, train_all_embed['label'])
y_true = test_all_embed['label']
y_pred = lr_model.predict(test_all_embed_features)

print('Topic-independent LR Test Accuracy: %.4f' % (accuracy_score(y_true, y_pred) * 100))
print('Topic-independent LR F1 Score: %.4f' % (f1_score(y_true, y_pred) * 100))
print('')

# Train and evaluate accuracy of LR classifier with topic-dependent (with fine-tuning) embedding features. 
lr_model = LR(penalty='l2', max_iter=1000, solver='lbfgs').fit(train_topic_embed_features_opt1, train_topic_embed_opt1['label'])
y_true = test_topic_embed_opt1['label']
y_pred = lr_model.predict(test_topic_embed_features_opt1)

print('Topic-dependent (with fine-tuning) LR Test Accuracy: %.4f' % (accuracy_score(y_true, y_pred) * 100))
print('Topic-dependent (with fine-tuning) LR F1 Score: %.4f' % (f1_score(y_true, y_pred) * 100))
print('')

# Train and evaluate accuracy of LR classifier with topic-dependent (without fine-tuning) embedding features. 
lr_model = LR(penalty='l2', max_iter=1000, solver='lbfgs').fit(train_topic_embed_features_opt2, train_topic_embed_opt2['label'])
y_true = test_topic_embed_opt2['label']
y_pred = lr_model.predict(test_topic_embed_features_opt2)

print('Topic-dependent (without fine-tuning) LR Test Accuracy: %.4f' % (accuracy_score(y_true, y_pred) * 100))
print('Topic-dependent (without fine-tuning) LR F1 Score: %.4f' % (f1_score(y_true, y_pred) * 100))
