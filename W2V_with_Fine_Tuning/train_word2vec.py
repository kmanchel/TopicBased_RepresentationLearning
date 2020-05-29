from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
import os.path
import glob
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

for i in range(40):
    for filename in glob.glob('data/datacorpus_word2vec_' + str(i) + '*'):
        os.remove(filename)

model = Word2Vec(LineSentence('data/datacorpus_all.txt'), size=300, window=5, min_count=1, workers=multiprocessing.cpu_count())
model.save('data/datacorpus_word2vec_all.model')
model.wv.save('data/datacorpus_word2vec_all_wv.model')

for i in range(40):
    if not os.path.isfile('data/datacorpus_' + str(i) + '.txt'):
        continue
    
    # Fine-tune W2V model.
    model = Word2Vec.load('data/datacorpus_word2vec_all.model')
    model.build_vocab(LineSentence('data/datacorpus_' + str(i) + '.txt'), update=True)
    total_examples = model.corpus_count

    model.train(LineSentence('data/datacorpus_' + str(i) + '.txt'), total_examples=total_examples, epochs=model.epochs)
    model.save('data/datacorpus_word2vec_' + str(i) + '_opt1.model')
    model.wv.save('data/datacorpus_word2vec_' + str(i) + '_wv_opt1.model')

    # Don't fine-tune W2V model.
    model = Word2Vec(LineSentence('data/datacorpus_' + str(i) + '.txt'), size=300, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save('data/datacorpus_word2vec_' + str(i) + '_opt2.model')
    model.wv.save('data/datacorpus_word2vec_' + str(i) + '_wv_opt2.model')
