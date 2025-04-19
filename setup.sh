#!/bin/bash
python -m nltk.downloader punkt wordnet brown
python -m textblob.download_corpora
