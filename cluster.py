import numpy as np
from collections import defaultdict, Counter
from itertools import chain
from operator import itemgetter
from tqdm import tqdm
from time import time   
from argparse import ArgumentParser

#### utils 

def seg_stem(word, adf, aftype):
    assert aftype in ["prefix", "suffix"]
    "Remove first prefix or last suffix"
    stem = word[len(adf):] if aftype == "prefix" else word[:-len(adf)]
    return stem

def seg_adfix(word, stem, aftype):
    assert aftype in ["prefix", "suffix"]
    "Remove stem"
    adfix = word[:-len(stem)] if aftype == "prefix" else word[len(stem):]
    return adfix

def join_adfix(word, adf, aftype):
    assert aftype in ["prefix", "suffix"]
    "Append prefix or suffix"
    new_word = adf + word if aftype == "prefix" else word + adf
    return new_word

def shared_seq(word1, word2, reverse=False):
    "Returns len of characters shared by two words"
    if reverse:
        word1, word2 = word1[::-1], word2[::-1]
    idx = 0
    while word1[idx] == word2[idx]:
        idx += 1
        if idx == len(word1) or idx == len(word2):
            break
    return idx


#### trie classes

class CharNode(object):

    def __init__(self, char, position=0, final=False):
        self.char = char
        self.pos = position
        self.children = {}
        self.is_word_final = final
        self.parent = None
        self.counter = 1
        self.seg_label = [] 
        self.seg_children = {}
        
    def add_child(self, child):
        if child.char not in self.children:
            self.children[child.char] = child
        
    def add_parent(self, parent):
        self.parent = parent
        
    def has_child(self, char):
        return char in self.children
    
    def get_child(self, char):
        return self.children[char]
    
    def incr_child(self, char):
        self.children[char].counter += 1
    
    def get_parent(self):
        return self.parent
    
    def set_word_final(self):
        self.is_word_final = True
    
    def dfs(self, str_accum):
        str_accum += self.char
        ls = []
        if self.is_word_final:
            ls.append(str_accum)
        if not self.children:
            return ls
        return ls + list(chain(*[self.get_child(char).dfs(str_accum) for char in self.children]))
    
    def get_child_strings(self):
        result = chain(*[self.get_child(char).dfs("") for char in self.children])
        return list(result)

class Trie(object):
    
    def __init__(self, reverse=False):
        self.root = CharNode("*", -1)
        self.vocab = Counter()
        self.reverse = reverse
        self.labels = ["suffix", "stem", "prefix"] if reverse else ["prefix", "stem", "suffix"]
        self.segs = defaultdict(lambda: defaultdict(list))
        self.segfreq = {}
        self.n_segs = 0
        self.w2i = []
        self.clusters = defaultdict(list) 
        self.cluster_map, self.cluster_scores = {}, {}
        self.cooccurences = []
        
    def has_word(self, word):
        return self.vocab[word] > 0
        
    def add_word(self, word):
        if self.has_word(word):
            self.vocab[word] += 1
            return
        self.vocab[word] += 1
        cn = self.root
        word = word[::-1] if self.reverse else word
        for i, char in enumerate(word):
            if not cn.has_child(char):
                cn.add_child(CharNode(char, i))
            else:
                cn.incr_child(char)
            cn = cn.get_child(char)
        cn.set_word_final()

    def set_segcount(self):
        self.n_segs = max(self.vocab.values())

    def add_seg(self, word, seg, addword=True):
        if addword:
            self.add_word(word)
        assert self.has_word(word)
        for label in self.labels:
            charspan = seg[label][::-1] if self.reverse else seg[label]
            charspan = charspan.split()
            for i in range(len(charspan)):
                seglabel = label + ":" + str(i)
                self.segs[seglabel][charspan[i][::-1] if self.reverse else charspan[i]].append(word)

    def index_vocab(self):
        if self.reverse:
            self.w2i = [w[::-1] for w in sorted([w[::-1] for w in self.vocab.keys()])]
        else:
            self.w2i = sorted(self.vocab.keys())

    # compare label frequency by segment form
    def most_common_labels(self):
        counts = []
        for label in self.segs:
            segments = Counter({k: len(v) for k, v in self.segs[label].items()}).most_common()
            counts.extend([(label, segment, count) for segment, count in segments])
        return sorted(counts, key=itemgetter(2), reverse=True)

    # determine prefixing v suffixing
    def most_common_frequent_label(self, topn=None, remove="stem"):
        if not topn:
            topn = int(np.ceil(np.log(len(self.vocab))))
        segment_counts = self.most_common_labels()
        most_freq_segs = Counter(map(lambda x: x[0].split(":")[0], segment_counts[:topn]))
        if remove:
            most_freq_segs[remove] = 0
        return most_freq_segs.most_common(1)[0][0]
    
    def _traverse_prefix(self, prefix):
        cn = self.root
        for char in start:
            cn = cn.get_child(char) 
        return cn
            
    def get_all_continuations(self, start):
        cn = self._traverse_prefix(start)
        return cn.get_child_strings()
    
    def score_continuations(self, start, label, segdict=None, norm=True, alpha=.2):
        aftype = label.split(":")[0]
        if not segdict:
            segdict = self.segs
        if self.reverse:
            start = start[::-1]
        cn = self.root
        for char in start:
            cn = cn.get_child(char)
        children = cn.get_child_strings()
        if label not in self.segfreq:
            if norm:
                # normalize to frequency per 10K words
                self.segfreq[label] = Counter({k: 10000 * len(v) / len(self.w2i) for k, v in segdict[label].items()})
            else:
                self.segfreq[label] = Counter({k: len(v) for k, v in segdict[label].items()})
        freq = self.segfreq[label] 
        scores = [] 
        for child in children:
            if self.reverse:
                child = child[::-1]
            scores.append(np.log(freq[child]) if child in freq else np.log(1+alpha))
            form = join_adfix(start[::-1], child[::-1], aftype) if self.reverse else join_adfix(start, child, aftype)
            count = Counter(segdict[label][child])[form]
            if norm:
                count /= self.n_segs
            scores[-1] *= np.sqrt(count + alpha)
        return zip(children, scores)

    def initial_cluster(self, label, segdict=None, segmentations=None, score_thresh=2, min_wordlen=2, verbose=False):
        clust_id = -1
        self.clusters = defaultdict(list) 
        self.cluster_map, self.cluster_scores = {}, {}
        self.cooccurences = []
        aftype, ind = label.split(":")
        for word in tqdm(self.w2i):
            if word in self.cluster_map:
                continue
            if verbose:
                print(word)
            clust_id += 1
            self.clusters[clust_id].append(word)
            self.cluster_map[word] = clust_id
            avgs = {}
            if len(word) < min_wordlen:
                continue
            continuations_bare = dict(self.score_continuations(word, label, segdict, norm=True))
            scores = np.array(list(continuations_bare.values()))
            if np.sum(scores >= score_thresh):
                avg = np.mean(scores[scores >= score_thresh])
                avgs[avg] = {"stem": word}
                avgs[avg]["words"] = {join_adfix(word, c, aftype): s for c, s in continuations_bare.items() if s >= score_thresh}
            if verbose:
                print(f'  c_bare clust_size: {len(scores) + 1}')
                print(f'  c_bare clust_size over thresh: {sum(scores >= score_thresh)}')
                print(f'    c_bare avg over thresh: {np.mean(scores[scores >= score_thresh])}')
            if segmentations:
                stems = set([seg_stem(word, seg[aftype], aftype) for seg in segmentations[word] if seg[aftype]])
                for stem in stems:
                    if len(stem) >= min_wordlen:
                        continuations_stem = dict(self.score_continuations(stem, label, segdict, norm=True))
                        scores = np.array(list(continuations_stem.values()))
                        if np.sum(scores >= score_thresh):
                            adfix = seg_adfix(word, stem, aftype)
                            if adfix in continuations_stem and continuations_stem[adfix] < score_thresh:
                                continue
                            if verbose:
                                print(f'  c_stem {stem} clust_size: {len(scores)}')
                                print(f'  c_stem {stem} clust_size over thresh: {sum(scores >= score_thresh)}')
                                print(f'    c_stem {stem} avg over thresh: {np.mean(scores[scores >= score_thresh])}')
                                print(f'    c_stem {stem} over thresh: {", ".join([c for c, s in continuations_stem.items() if s >= score_thresh])}')
                            avg = np.mean(scores[scores >= score_thresh])
                            avgs[avg] = {"stem": stem}
                            avgs[avg]["words"] = {join_adfix(stem, c, aftype): s for c, s in continuations_stem.items() if s >= score_thresh}
            if not avgs:
                self.cooccurences.append(("", word))
                continue
            max_avg_score = np.max(list(avgs.keys()))
            if verbose:
                print(max_avg_score)
            if max_avg_score:
                stem = avgs[max_avg_score]["stem"]
                self.cluster_scores[stem] = {}
                if word == stem:
                    self.cooccurences.append(("", word))
                    self.cluster_scores[stem][word] = 0
                for clword in avgs[max_avg_score]["words"]:
                    if clword not in self.cluster_map:
                        self.cluster_map[clword] = clust_id
                        self.clusters[clust_id].append(clword)    
                        self.cooccurences.append((seg_adfix(clword, stem, aftype), stem))   
                        self.cluster_scores[stem][clword] = avgs[max_avg_score]["words"][clword] 
                    elif clword == word and word != stem:
                        self.cooccurences.append((seg_adfix(clword, stem, aftype), stem))
                        self.cluster_scores[stem][word] = 0
        
    def write_cluster_predictions(self, outpath):
        with open(outpath, 'w') as outfile:
            for clust in self.clusters:
                outfile.write("\n".join(self.clusters[clust])+"\n\n")

    def write_cluster_scores(self, outpath):
        with open(outpath, 'w') as outfile:
            outfile.write("stem,word,score\n")
            for stem in self.cluster_scores:
                for word, score in self.cluster_scores[stem].items():
                    outfile.write(f'{stem},{word},{score}\n')

def aggregate_segmentations(lang, kwargs):
    segmentations = {"all": defaultdict(list)}
    for g in kwargs.grammars.split():
        gram = {}
        gram["all"] = defaultdict(list)
        for r in range(kwargs.n):
            gram[r] = {}
            seg3data = f"{kwargs.segdir}{lang}.G{g}.R{r}.seg.csv"
            with open(seg3data, 'r') as seg:
                for i, line in enumerate(seg):
                    if i == 0:
                        header = line.strip().split(",") # word,prefix,stem,suffix
                        continue
                    fields = line.strip().split(",")
                    wordseg = dict(zip(header[1:], fields[1:]))
                    gram[r][fields[0]] = wordseg
                    # gather segmentations across runs
                    gram["all"][fields[0]].append(wordseg)
                    segmentations["all"][fields[0]].append(wordseg)
        segmentations[g] = gram
    return segmentations

def build_tries(segmentations):
    """Store vocabulary in two tries, one forward, one reverse, 
    to get peripheral segments in both directions. """

    trieF, trieR = Trie(), Trie(reverse=True)

    for trie in [trieF, trieR]:
        for word in tqdm(segmentations):
            for seg in segmentations[word]:
                trie.add_seg(word, seg)
        trie.index_vocab()
        trie.set_segcount()

    # determine prefixing vs. suffixing
    assert trieF.most_common_frequent_label() == trieR.most_common_frequent_label()
    adfix_direction = trieF.most_common_frequent_label()
    return [trieF, trieR, adfix_direction]

if __name__ == '__main__':
    parser = ArgumentParser(description='Cluster words into paradigms from Adaptor Grammar segmentations.')
    parser.add_argument('--segdir', help='Data directory (*.seg.csv will be read from here)',
        default="2021Task2/predictions/MorphAGram/")
    parser.add_argument('--preddir', help='Directory to write predictions', 
        default="2021Task2/predictions/")
    parser.add_argument('--langs', help='Languages to read, space-separated string', 
        default="Swedish Portuguese Maltese Persian Russian Basque Bulgarian English Finnish German Kannada Navajo Spanish Turkish")
    parser.add_argument('--grammars', help='Indices of grammars used, space-separated string', default="1 4")
    parser.add_argument('--scores', help='Score thresholds for initial cluster, space-separated string', 
        default="2") 
    parser.add_argument('--n', help='Number of sampler runs per lang and grammar', default=3)
    args = parser.parse_args()

    for lang in args.langs.split():
        print(lang)
        segmentations = aggregate_segmentations(lang, args)
        trieF, trieR, inflection_direction = build_tries(segmentations["all"])
        searchTrie = trieR if inflection_direction == "prefix" else trieF
        segTrie = trieF if inflection_direction == "prefix" else trieR
        for score_thresh in args.scores.split():
            searchTrie.initial_cluster(inflection_direction+":0", segTrie.segs, segmentations["all"], float(score_thresh))
            searchTrie.write_cluster_predictions(f'{args.preddir}{lang}.T{score_thresh.replace(".", "_")}.preds')

