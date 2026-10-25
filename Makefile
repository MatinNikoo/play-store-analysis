# make all     rebuilds every number and figure from the raw data (about 1.5 hours on 2 cores)
# make test    runs the tests on a small fake dataset (a few minutes, no download)
# The live simulator in simulator/ is never rebuilt here, on purpose (it's live on my site).

PY ?= python3
RUN = OMP_NUM_THREADS=2 nice -n 10 $(PY) -u

.PHONY: all data analysis study paper test clean

all: analysis study paper

data: data/google_playstore_2021.csv

data/google_playstore_2021.csv:
	bash get_data.sh

analysis: data
	$(RUN) analysis.py

study: data
	$(RUN) -m study.prepare
	$(RUN) -m study.ranking
	$(RUN) -m study.milestones
	$(RUN) -m study.checks
	$(RUN) -m study.market
	$(RUN) -m study.associations

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -quiet paper.tex

test:
	$(PY) -m pytest -q

clean:
	rm -f data/clean.pkl data/study.pkl data/study_features.pkl data/study_features_ranking_preds.pkl
	rm -f data/milestone_input.pkl data/associations_input.pkl data/market_input.pkl
	cd paper && latexmk -C -quiet paper.tex || true
