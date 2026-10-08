#!/usr/bin/env bash
# Downloads the 2.3M app Google Play dataset (June 2021, MIT license, by Gautham Prakash)
# and combines it into data/google_playstore_2021.csv (about 650 MB, so it isn't stored in this repo).
set -e
mkdir -p data
git clone --depth 1 https://github.com/gauthamp10/Google-Playstore-Dataset.git data/source
cd data/source/dataset
for f in *.tar.gz; do tar -xzf "$f"; done
# Part1 has the header row, Part2 and Part3 don't, so just stack them
cat Part1.csv Part2.csv Part3.csv > ../../google_playstore_2021.csv
cd ../../..
rm -rf data/source
echo "Saved data/google_playstore_2021.csv"
