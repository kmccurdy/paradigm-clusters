import subprocess as sp
from datetime import datetime
from collections import defaultdict

from argparse import ArgumentParser

import sys
sys.path.append('MorphAGram/')
from preprocessing import *
from segmentation import *

# reproduce AG sampler settings from MorphAGram paper
pyags = "py-cfg/py-cfg -r 0 -d 10 -x 10 -D -E -e 1 -f 1 -g 10 -h 0.1 -w 1 -T 1 -m 0 -n 500 -N 1 "

# train grammars 1 and 4 from MorphAGram paper

grammars = {1: {"prefix_nonterminal": "PrefixMorph", 
				"stem_nonterminal": "Stem", 
				"suffix_nonterminal": "SuffixMorph"},
			4: {"prefix_nonterminal": "Prefix", 
				"stem_nonterminal": "Stem", 
				"suffix_nonterminal": "Suffix"}
				}

def preprocess_lex(lex_path, outpath):
	"Preprocess lexical data for PYAGS, following README"
	# Read the initial lexicon (word list), and convert it into Hex.
	words, encoded_words, hex_chars = process_words(lex_path)
	write_encoded_words(encoded_words, outpath+".enc.lex")
	return hex_chars

def preprocess_gram(hex_chars, gram_path, outpath):
	"Preprocess grammar for PYAGS, following README"
	# Read the initial CFG and append the HEX encoded characters as terminals.
	grammar = read_grammar(gram_path)
	appended_grammar = add_chars_to_grammar(grammar, hex_chars)
	write_grammar(appended_grammar, outpath+".cfg")

def run_pyags(lex_path, gram_path, outpath):
	"Call py-cfg with appropriate settings"
	#outfiles = f"-G {outpath}.pycfg.cfg -A {outpath}.pycfg.parse -F {outpath}.pycfg.trace "
	#infiles = f"{outpath}.cfg < {outpath}.enc.lex"
	file_args = f"-G {outpath}.pycfg.cfg -A {outpath}.pycfg.parse -F {outpath}.pycfg.trace {gram_path}.cfg "
	args = pyags + file_args #outfiles + infiles
	#sp.run(args.split())
	with open(f"{lex_path}.enc.lex", 'r') as lex:
		sp.call(args.split(), stdin=lex)

def seg_pyags(outpath, grammar_kwargs):
	"""Parse segmented text, returns model."""
	return parse_segmentation_output(outpath+".pycfg.parse", 
		normalized_segmentation_output_path=outpath+".pycfg.seg",
		**grammar_kwargs)

def seg_adfix(word, seg, aftype):
    assert aftype in ["prefix", "suffix"]
    "Segment first prefix or last suffix, as most likely inflectional morphology candidate"
    adfix = seg[0] if aftype == "prefix" else seg[-1]
    rest = word[len(adfix):] if aftype == "prefix" else word[:-len(adfix)]
    return (adfix, rest)

# TODO: need a more sophisticated strategy here for handling grammars w/o prefix-stem distinction
def write_seg_clusters(seg_dict, outpath, grammar_kwargs):
	"""Predict paradigm clusters from segmentation.
	Writes three files:
		- *.stem.preds - predicted clusters based on stem alone
		- *.prefixstem.preds - predicted clusters based on prefix + stem
		- *.seg.csv - segmentation for each word in csv format
	"""
	#stemNT = grammar_kwargs["stem_nonterminal"]
	#prefixNT = grammar_kwargs["prefix_nonterminal"]
	stemNT, prefixNT, suffixNT = "stem", "prefix", "suffix"
	# preliminary clustering method (section 3.2. of SIGMORPHON 2021 paper):
	# cluster by prefix + stem, or stem alone
	clust = {"stem": defaultdict(list), "prefixstem": defaultdict(list)}
	# add clustering for finer-grained segmentations
	wo_adfx_clust = {prefixNT: defaultdict(list), suffixNT: defaultdict(list)} 
	for grouping in clust:
		with open(f"{outpath}.{grouping}.preds", "w") as clustfile:
			for group in clust[grouping]:
				clustfile.write("\n".join(clust[grouping][group])+"\n\n")
	for grouping in wo_adfx_clust:
		with open(f"{outpath}.wo_{grouping}.preds", "w") as clustfile:
			for group in wo_adfx_clust[grouping]:
				clustfile.write("\n".join(wo_adfx_clust[grouping][group])+"\n\n")
	# write csv file of segmentations directly -
	# input to later clustering algorithm (sect. 3.4 of SIGMORPHON 2021 paper)
	with open(f"{outpath}.seg.csv", "w") as segfile:
		segfile.write("word,prefix,stem,suffix\n")
		for word, seg in seg_dict.items():
			segfile.write(f"{word},{seg[prefixNT]},{seg[stemNT]},{seg[suffixNT]}\n")
			clust["stem"][seg[stemNT]].append(word)
			clust["prefixstem"][seg[prefixNT] + seg[stemNT]].append(word)
			for adfxNT in wo_adfx_clust:
				seg_input = seg[adfxNT].split() if " " in seg[adfxNT] else [seg[prefixNT], seg[stemNT], seg[suffixNT]]
				adfix, rest = seg_adfix(word, seg_input, adfxNT)
				wo_adfx_clust[adfxNT][rest].append(word)

if __name__ == '__main__':
	parser = ArgumentParser(description='Train standard MorphAGram on S21T2 data.')
	parser.add_argument('--lexdir', help='Data directory (lexicon *.lex will be read from here)',
		#default="2021Task2/data/dev_langs/")
		default="2021Task2/data/test_langs/")
	parser.add_argument('--cfgdir', help='Path to directory w grammar*.txt files',
		default="MorphAGram/data/georgian/grammar/standard/")
	parser.add_argument('--outdir', help='Directory to write output', 
		default="MorphAGram/data/S21T2/")
	parser.add_argument('--segdir', help='Directory to write segmentations', 
		default="2021Task2/predictions/MorphAGram/")
	parser.add_argument('--langs', help='Languages to read, space-separated string', 
		#default="Swedish Portuguese Maltese Persian Russian")
		default="Basque Bulgarian English Finnish German Kannada Navajo Spanish Turkish")
	parser.add_argument('--n', help='Number of sampler runs per lang and grammar', default=3)
	parser.add_argument('--train', help='Flag: train model', action="store_true")
	args = parser.parse_args()

	for lang in args.langs.split():
		lpath = f"{args.outdir}{lang}"
		hex_chars = preprocess_lex(f"{args.lexdir}{lang}.lex", lpath)
		for grammar in grammars:
			lgpath = lpath + f".G{grammar}"
			preprocess_gram(hex_chars, f"{args.cfgdir}grammar{grammar}.txt", lgpath)
			print(f"Processing {lang}, grammar {grammar}")
			for r in range(1, int(args.n)+1):
				print(f"  Run {r} -- {datetime.now()}")
				rpath = lgpath + f".R{r}"
				while args.train:
					run_pyags(lpath, lgpath, rpath)
					break
				seg_model = seg_pyags(rpath, grammars[grammar])
				ppath = f"{args.segdir}{lang}.G{grammar}.R{r}" 
				write_seg_clusters(seg_model[0], ppath, grammars[grammar])

