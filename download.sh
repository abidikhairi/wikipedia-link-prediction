#!/bin/bash

echo "Starting download of Wikipedia link prediction datasets and pre-trained word embeddings"

TRAINING_EDGES=https://github.com/abidikhairi/wikipedia-link-prediction/raw/main/data/train.csv
TEST_EDGES=https://github.com/abidikhairi/wikipedia-link-prediction/raw/main/data/test.csv
WORD2VEC_FEATURES=1Qm54SOKmmWy066s90EjQw5hF3t9Hv8dE

if [ ! -d "data" ]; then
  echo "Creating data directory"
  mkdir data
fi

echo "Downloading pre-trained word embeddings"
cd data 
gdown "$WORD2VEC_FEATURES"

echo "Downloading training and test datasets"
wget "$TRAINING_EDGES"
wget "$TEST_EDGES"

echo "Download complete!"
