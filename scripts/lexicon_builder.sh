#!/bin/bash

words=$(cat data/glove.6B/glove.6B.50d.txt | cut -d' ' -f1)
num_words=$(echo "$words" | wc -l)
echo $num_words > data/glove.6B/words.txt
echo "$words" >> data/glove.6B/words.txt

chars=$(echo "$words" | sed 's/./\0\n/g' - | sort -u)
num_chars=$(echo "$chars" | wc -l)
echo $num_chars > data/glove.6B/chars.txt
echo "$chars" >> data/glove.6B/chars.txt
