Code for [Multidomain Language Models for Green NLP](https://www.aclweb.org/anthology/2021.adaptnlp-1.1/).

## RAW TEXT DOMAIN DATA

* [Amazon Reviews](https://nijianmo.github.io/amazon/index.html)
* [Arxiv Papers](https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view?usp=sharing)
* [Realnews](https://github.com/rowanz/grover/blob/master/realnews/realnews_tiny.jsonl)
* [Reddit Comments](https://huggingface.co/datasets/reddit)


## SUPERVISED TASK DATA

* [ACL-ARC](https://web.eecs.umich.edu/~lahiri/acl_arc.html)
* [AG-News](https://huggingface.co/datasets/ag_news)
* [ChemProt](http://potentia.cbs.dtu.dk/ChemProt/)
* [Clothing Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)
* [HyperPartisan](https://huggingface.co/datasets/hyperpartisan_news_detection)
* [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/)
* [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)
* [PubMed-RCT](https://github.com/Franck-Dernoncourt/pubmed-rct)
* [SARC](https://nlp.cs.princeton.edu/SARC/2.0/)
* [SciCite](https://huggingface.co/datasets/scicite)
* [TalkDown](https://github.com/zijwang/talkdown)


## CODE

Code is split in multiple evaluation files, one for each task. Models are not provided, but can be pretrained separately using the `run_language_modeling.py` script provided here (or by [HuggingFace](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)).

Each script is indicative of the code run in our machines. Train/dev/test splits are not provided, as they were randomly sampled. Nevertheless, the scripts were tested with multiple samples and performance was as similar as it can be to reported results.
