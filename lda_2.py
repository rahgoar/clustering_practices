# -*- coding: utf-8 -*-
"""LDA_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11xht92-hZFg1qc6F-KcmCe3SQ3a4PTPo
"""

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'This is the first document.',
'This document is the second document.',
'And this is the third one.',
'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())

print(X2.toarray())

from collections import defaultdict
from gensim import corpora

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

from gensim import models

tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi_model.print_topics(2)

# both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
for doc, as_text in zip(corpus_lsi, documents):
    print(doc, as_text)

#Model persistency is achieved with the save() and load() functions

lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
corpus_lda = lda_model[corpus_tfidf]
lda_model.print_topics(2)

for doc, as_text in zip(corpus_lda, documents):
    print(doc, as_text)
