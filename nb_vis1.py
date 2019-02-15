from gensim import corpora, models, similarities
import codecs
import json
import pyLDAvis
import pyLDAvis.gensim
with codecs.open("../input/hafez_Train3cls_cls3.txt", "r", 'UTF-8') as myfile:
    documents=myfile.readlines()

with codecs.open("../../stop-words_persian_1_fa.txt","r", 'UTF-8') as myfile:
	stoplist=myfile.read()
#textha = [[word for word in document.lower().split() if word not in stoplist]
          #for document in matns]
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]
 # remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#textha = [[word for word in text if word not in tokens_once]
 #         for text in textha]
texts = [[word for word in text if word not in tokens_once]
          for text in texts]
#loghatname = corpora.Dictionary(textha)
dictionary = corpora.Dictionary(texts)
#maincorpus = [loghatname.doc2bow(text) for text in textha]
corpus = [dictionary.doc2bow(text) for text in texts]
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=20,passes=10)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda,corpus,dictionary)
vis
