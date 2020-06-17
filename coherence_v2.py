# -*- coding: UTF-8 -*-
from __future__ import print_function
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
from gensim.sklearn_api import TfIdfTransformer
import codecs
import numpy as np
from hazm import *
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def calc_coh(infile):
    with codecs.open(infile, "r", 'UTF-8') as myfile:
        documents=myfile.readlines()
    with codecs.open("../../stop-words_persian_1_fa.txt","r", 'UTF-8') as myfile:
        stoplist=myfile.read()
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once] for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=20, passes=3)
    cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    cm2 = CoherenceModel(model=lda, texts=texts, coherence='c_v')
    coherence = cm.get_coherence()  # get coherence value
    coherence_v = cm2.get_coherence()
    perp = lda.log_perplexity(corpus)
    return coherence, coherence_v, perp


classes = 3 
kpis = []
for cls in range(1, classes+1):
    hfz_file = "../input/hafez_Train6cls_cls" + str(cls) + ".txt"
    kpis.append(calc_coh(hfz_file))

print(kpis)
print(np.average(kpis,axis=0))
