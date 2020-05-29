Dataset corpus experiments without fine-tuning: see W2V_without_Fine_Tuning/dataset_corpus.py
---------------------------------------------------------------------------------------------
Arguments (all optional):
  --dataset DATASET     Dataset for task, 1 for irony, 2 for nepal disaster, 3 for bias (1 as default)
  --topic TOPIC         Topic modeling option, 1 for lda, 2 for nmf (1 as default)
  --numtopics NUMTOPICS Number of topics (5 as default)
  --mincount MINCOUNT   Min_count for w2v (5 as default)
  --worddim WORDDIM     Word embedding dimension (300 as default)
Datasets used: bias_only_3k.csv (first 3000 entries of the Bias dataset), irony.csv (Irony Dataset), preprocessed_nepal.csv (Nepal Dataset). Make sure they are in the same directory as the script.
Note: make sure folders 'irony_tmp', 'nepal_tmp', 'partisan_tmp' are in the same directory as the script and are empty before each experiment.

Dataset corpus experiments with and without fine-tuning: see W2V_with_Fine_Tuning/
----------------------------------------------------------------------------------
Files:
- train_lda.py
  + Trains LDA topic model
- process_wiki.py
  + Splits dataset by topics
- train_word2vec.py
  + Trains Word2Vec model on entire dataset (topic-independent)
  + Trains Word2Vec model on each topic (with fine-tuning on the topic-independent Word2Vec model)
  + Trains Word2Vec model on each topic (from scratch/without fine-tuning)
- embed_all.py
  + Processes dataset by embedding the documents using the topic-independent Word2Vec model
- embed_topics.py
  + Processes dataset by embedding the documents using topic-dependent Word2Vec models (with fine-tuning)
  + Processes dataset by embedding the documents using topic-dependent Word2Vec models (without fine-tuning)
- lr.py
  + Trains logistic regression classifier with (1) topic-independent embeddings, (2) fine-tuned topic-dependent embeddings,
    and (3) topic-dependent embeddings trained from scratch
  + Reports test accuracy and F1 score for each embedding type

IMPORTANT: 
All data files can be found at (due to large size): https://github.com/ArtfulBottom/NLP-Project/tree/master/src/data

The data files are:
- irony.csv
  + Preprocessed irony classification dataset
- preprocessed_nepal.csv
  + Preprocessed Nepal disaster classification dataset
- preprocessed_queensland.csv
  + Preprocessed Queensland disaster classification dataset
- bias_only_3k.csv
  + Preprocessed Partisan bias classification dataset
- datacorpus_embed_all*
  + Corresponds to the topic-independent embedding features obtained on each dataset
- datacorpus_embed_topics_opt1*
  + Corresponds to the topic-dependent embedding features (with fine-tuning) obtained on each dataset
- datacorpus_embed_topics_opt2*
  + Corresponds to the topic-dependent embedding features (without fine-tuning) obtained on each dataset

For each dataset, we used the following tuned hyperparameters:
- irony.csv
  + Number of topics: 20, Word2Vec mincount: 1, default Word2Vec options otherwise
- preprocessed_nepal.csv
  + Number of topics: 5, Word2Vec mincount: 1, default Word2Vec options otherwise
- preprocessed_queensland.csv
  + Number of topics: 5, Word2Vec mincount: 1, topic-dependent fine-tuning epochs: 20, default Word2Vec options otherwise
- bias_only_3k.csv
  + Number of topics: 10, Word2Vec mincount: 1, default Word2Vec options otherwise

BERT Topic Based Finetuning:
----------------
Running these experiments will require a GPU/TPU. 
The full implementation, results, and discussions are recorded in "TopicBased_BERT_Finetuning.ipynb" such that results are visible without having to run the code. Alternatively, "TopicBased_BERT_Finetuning.py" runs the experiments if a GPU is present (~4 hours to run).
Finetuning is run for all 4 datasets. Links for dataset also provided in notebook but they will need to be manually downloaded due to size limit:
- Nepal Dataset: https://drive.google.com/open?id=1uYgcnTEZ5YEAdB6rLdWhHa7YuYJh9Wz3 
- Queensland Dataset: https://drive.google.com/open?id=1IbqkpjHk_lzqUgnPxAjPifTbECmZ3p1P
- Irony Dataset: https://drive.google.com/file/d/1yCULJV5EDc_EoHcNTxP8ovZ903KxJtYV/view?usp=sharing 
- Hyperpartisan Bias Dataset: https://drive.google.com/open?id=1k4cSuS1Ww2U92CnEckVIh_QqF3JUkIsO 

**Note that the Hyperpartisan Bias Dataset has been already preprocessed from a much larger xml file. For original preprocessing scripts, please visit: https://github.com/kmanchel/cs577_project/tree/master/src 
