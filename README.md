# paradigm-clusters

Code to generate paradigm clusters for the [2021 SIGMORPHON Shared Task 2](https://github.com/sigmorphon/2021Task2).

Disclaimer: everything here is extremely hacky research code. Caveat emptor.

Has only been run on Linux machines so far, but should work on OSX as well if you can compile the PY-AGS sampler.

# Setup

Run the setup script:

```
sh setup.sh
```

Requirements:

- Conda
- wget 
- tar
- git
- g++ (to compile the py-ags sampler)

# Run

To run on dev or test data:

```
sh run.sh dev
sh run.sh test
```

# Cite

If you use or reference this code in your work, please cite the paper:

```
@inproceedings{mccurdy-etal-2021-adaptor,
    title = "{A}daptor {G}rammars for Unsupervised Paradigm Clustering",
    author = "McCurdy, Kate  and
      Goldwater, Sharon  and
      Lopez, Adam",
    booktitle = "Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.sigmorphon-1.9",
    doi = "10.18653/v1/2021.sigmorphon-1.9",
    pages = "82--89",
    abstract = "This work describes the Edinburgh submission to the SIGMORPHON 2021 Shared Task 2 on unsupervised morphological paradigm clustering. Given raw text input, the task was to assign each token to a cluster with other tokens from the same paradigm. We use Adaptor Grammar segmentations combined with frequency-based heuristics to predict paradigm clusters. Our system achieved the highest average F1 score across 9 test languages, placing first out of 15 submissions.",
}
```

