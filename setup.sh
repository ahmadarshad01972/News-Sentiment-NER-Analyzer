#!/bin/bash

# Download necessary NLTK corpora
python -m nltk.downloader punkt wordnet brown

# Download required corpora for TextBlob
python -m textblob.download_corpora

