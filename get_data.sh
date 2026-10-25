#!/usr/bin/env bash
# Downloads the 2.3M app Google Play dataset (June 2021, MIT license, by Gautham Prakash)
# and combines it into data/google_playstore_2021.csv (about 650 MB, so it isn't stored in this repo).
# The commit and the checksum of the combined file are pinned, so everyone gets the same bytes.
set -e
COMMIT=423196c20029a69d6bc68f1eafc3342ec79dcf8a
SHA256=f8830426d88687801d5874dfd1e1dfc209a1324e56eb837ab46e942dc4d328e1
mkdir -p data
git init -q data/source
cd data/source
git fetch -q --depth 1 https://github.com/gauthamp10/Google-Playstore-Dataset.git "$COMMIT"
git checkout -q FETCH_HEAD
cd dataset
for f in *.tar.gz; do tar -xzf "$f"; done
# Part1 has the header row, Part2 and Part3 don't, so just stack them
cat Part1.csv Part2.csv Part3.csv > ../../google_playstore_2021.csv
cd ../../..
rm -rf data/source
echo "$SHA256  data/google_playstore_2021.csv" | sha256sum -c -
echo "Saved data/google_playstore_2021.csv"
