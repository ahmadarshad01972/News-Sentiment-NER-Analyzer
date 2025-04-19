#!/bin/bash
python -m nltk.downloader punkt wordnet
python -m textblob.download_corpora
python -m spacy download en_core_web_sm
