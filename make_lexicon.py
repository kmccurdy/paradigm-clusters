# build lexicon from input data - format for MorphAGram
from collections import Counter
from argparse import ArgumentParser
from unicodedata import category
import sys

class TextProcesser(object):
	"Iterable to process data on read-in"

	def __init__(self, filename, writefile=None):
		self.filename = filename
		self.writefile = writefile
		self.punctuation_chars =  [chr(i) for i in range(sys.maxunicode) if (category(chr(i)).startswith("P") or category(chr(i)).startswith("S"))]
		#self.split_chars = ["-", "—", "—"]
		# N.B. ^ this targets Portuguese reflexive verbs, e.g. enterra-se
		# option here to split on more punc, e.g. apostrophes, tho Maltese is worry there
		#self.sentence_punct = [".", ",", "!", "?", ";", ":"] 
		# ^ mostly to handle Navajo
		self.split_chars = ["-", "—", "—", ".", ",", "!", "?", ";", ":", "[", "]"]

	def __iter__(self):
		if self.writefile:
			f = open(self.writefile, 'w')
		for line in open(self.filename, 'r'):
			ll = []
			for w in line.lower().split():
				if w not in self.punctuation_chars:
					for split in self.split_chars:
						w = w.replace(split, " ")
					ll.extend([w1 for w1 in w.split() if w1])
			if self.writefile:
				f.write(" ".join(ll) + "\n")
			if ll:
				yield ll
		if self.writefile:
			f.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='Process text data, create lexicon with frequency counts.')
    parser.add_argument('--datadir', help='Data directory (lexicon will be written here)',
    	default="2021Task2/data/dev_langs/")
    	#default="2021Task2/data/test_langs/")
    parser.add_argument('--lang', help='Language(s, comma-separated) to read', 
    	default="Maltese,Persian,Portuguese,Swedish,Russian")
    	#default="Basque,Bulgarian,English,Finnish,German,Kannada,Navajo,Spanish,Turkish")
    parser.add_argument('--write', help="Additionally write processed version of original data file",
    	action="store_true")
    args = parser.parse_args()

    for lang in args.lang.split(","):
    	print(f'Processing {lang}')
    	corpus = Counter() 
    	infile = args.datadir + lang + ".bible.txt"
    	if args.write:
    		reader = TextProcesser(infile, args.datadir + lang + ".bible.processed.txt")
    	else:
    		reader = TextProcesser(infile)
    	for line in reader:
    		corpus.update(line)
    	with open(args.datadir + lang + ".lexcnt", 'w') as cntlex, \
    		open(args.datadir + lang + ".lex", 'w') as lex:
    		for word, count in corpus.most_common():
    			cntlex.write(f"{word.lower()}\t{count}\n")
    			lex.write(f"{word.lower()}\n")
