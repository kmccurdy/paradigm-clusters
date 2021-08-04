#! /usr/bin/env bash

[[ $1 == "dev" || $1 == "test" ]] || { echo "Please specify 'dev' or 'test' to select data." >&2; exit 1; }

echo "Running paradigm clusters for $1 data"
echo

# activate environment if needed:
#conda activate paradigm-clusters

DPATH="2021Task2/data/$1_langs/"
langs="Maltese Persian Portuguese Russian Swedish"
langs="Maltese Persian"
if [ "$1" == "test" ]; then
	langs="Basque Bulgarian English Finnish German Kannada Navajo Spanish Turkish"
fi

MPATH=MorphAGram/data/S21T2/
PPATH=2021Task2/predictions/
MSEGS=$PPATH"MorphAGram/"

mkdir -p $MPATH
mkdir -p $MSEGS

T=2 # defaults to score threshold hyperparameter T=2, following paper
N=3 # defaults to 3 AG samples, following paper

for lang in $langs; do
	echo $lang

	# prepare the data
	python make_lexicon.py --datadir $DPATH --lang $lang

	# train & sample segmentations from Adaptor Grammar
	python run_pyags_morphagram.py --lexdir $DPATH --langs $lang --n $N --outdir $MPATH --segdir $MSEGS --train

	# cluster based on segmentations
	python cluster.py --langs $lang --scores $T --n $N

	# optional - evaluate cluster output vs. gold standard
	python 2021Task2/evaluate/eval.py --reference $DPATH$lang"."$1".gold" --prediction $PPATH$lang".T"$T".preds"

done

