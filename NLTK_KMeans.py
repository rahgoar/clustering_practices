from __future__ import print_function, unicode_literals, division

import copy
import random
import sys
from gensim import corpora, models, matutils
import numpy as np
import scipy
import codecs
from time import time
from sklearn import metrics
from scipy.stats import spearmanr

from nltk.cluster.util import VectorSpaceClusterer
from nltk.compat import python_2_unicode_compatible

def preprocess_1(fname):
        
        
    #docs = [(doc.strip()).split()[1:] for doc in codecs.open(fname, 'r', 'UTF-8')]
    with codecs.open(fname, "r", 'UTF-8') as myfile:
        docs=myfile.readlines()
        #cPickle.dump(docs, open(self.conf['fname_docs'], 'wb'))
    with codecs.open("../../stop-words_persian_1_fa.txt","r", 'UTF-8') as myfile:
        stoplist=myfile.read()
    #texts = [[word for word in document if word not in stoplist]
    texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in docs]

 # remove words that appear only once
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
          for text in texts]
    dictionary = corpora.Dictionary(texts)
        #dictionary.save(self.conf['fname_dict'])

    corpus = [dictionary.doc2bow(doc) for doc in texts]
        #corpora.MmCorpus.serialize(self.conf['fname_corpus'], corpus)
    tfidf1 = models.TfidfModel(corpus)
    corpus_tfidf = tfidf1[corpus]
        #return docs, dictionary, corpus_tfidf
    return docs, dictionary, corpus

def get_vectors(corpus):

        def get_max_id():
            maxid = -1
            for document in corpus:
                maxid = max(maxid, max([-1] + [fieldid for fieldid, _ in document])) # [-1] to avoid exceptions from max(empty)
            return maxid

        num_features = 1 + get_max_id()
        index = np.empty(shape=(len(corpus), num_features), dtype=np.float32)
        for docno, vector in enumerate(corpus):
            if docno % 1000 == 0:
                print("PROGRESS: at document #%i/%i" % (docno, len(corpus)))

            if isinstance(vector, np.ndarray):
                pass
            elif scipy.sparse.issparse(vector):
                vector = vector.toarray().flatten()
            else:
                vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
            index[docno] = vector        

        return index
def cluster3(index,k):
    from nltk.cluster import GAAClusterer
    clusterer = GAAClusterer(k)
    clusters = clusterer.cluster(index, True)
    return clusters

def cluster2(index,k):
    from sklearn.cluster import k_means
    from sklearn.cluster import KMeans
    cluster_center, result, inertia = k_means(index, n_clusters=k, init="k-means++")
    estimator = KMeans(init='k-means++', n_clusters=k, n_init=10)
    estimator.fit(index)
    return estimator.labels_

def cluster(index,k):
    # example from figure 14.9, page 517, Manning and Schutze

    from nltk.cluster import KMeansClusterer, euclidean_distance

    

    #vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

    # test k-means using the euclidean distance metric, 2 means and repeat
    # clustering 10 times with random seeds

    clusterer = KMeansClusterer(k, euclidean_distance, repeats=10)
    clusters = clusterer.cluster(index, True)
    print('Clustered:', index)
    print('As:', clusters)
    print('Means:', clusterer.means())
    print()
    return clusters

def main():
    data_dir ='../input'
    fname = data_dir + '/hafez_train_3cls_copy.csv'
    name='lda_tfidf'
    t0 = time()
    inertia = 1.1
    labels=[]
    i=0
    delimiter=','
    for line in codecs.open(fname, 'r', 'UTF-8'):
            row = line.split(delimiter)
            labels.append(row[1])
            
    with codecs.open('../output/labels.txt', 'w', 'UTF-8') as fo:
        for Y in labels:
            fo.write(Y)
    
    num_classes=3
    docs, dictionary, corpus = preprocess_1(fname)
    index = get_vectors(corpus)
    #clusters= cluster(index, k=num_classes)
    #clusters= cluster2(index, k=num_classes)
    clusters= cluster3(index, k=num_classes)
    print(len(clusters))
    print(len(labels))
    labels = list(map(int,labels))
    clusters = list(map(int,clusters))
    
    print(82 * '_')
    print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tkappa\tcorr\tsilh_Clus\tsilh_HMN')
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%-9s\t%.3f\t%.3f'
          % (name, (time() - t0), 
             metrics.homogeneity_score(labels, clusters),
             metrics.completeness_score(labels, clusters),
             metrics.v_measure_score(labels, clusters),
             metrics.adjusted_rand_score(labels, clusters),
             metrics.adjusted_mutual_info_score(labels,  clusters),
             metrics.cohen_kappa_score(labels, clusters,weights='linear'),
             str(spearmanr(labels,clusters)),
             metrics.silhouette_score(index, clusters,
                                      metric='euclidean'),
             metrics.silhouette_score(index, labels,
                                      metric='euclidean'),
             ))
    #corr=spearmanr(labels,clusters) )
'''metrics.cohen_kappa_score(labels, str(estimator.labels_),weights='linear'),'''

if __name__ == '__main__':
    main()
    