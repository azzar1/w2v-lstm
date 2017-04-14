#!/bin/bash
TMP=$(mktemp -d)
wget http://nlp.stanford.edu/data/glove.6B.zip -P ${TMP}
mkdir -p data/glove.6B
unzip ${TMP}/glove.6B.zip -d data/glove.6B
