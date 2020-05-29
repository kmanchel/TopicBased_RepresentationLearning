#!/bin/bash

#PBS -N preprocess_cleanAF
echo Starting Preprocessing
cd ~/NLP/src
source activate nlp
python preprocessing.py --set training --out ExtraClean_ --lower=1 --extra=1 --dir /home/u40332/NLP/data/bias
echo Finished Preprocessing
