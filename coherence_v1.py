from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

model = LdaModel(common_corpus, 5, common_dictionary)

cm = CoherenceModel(model=model, corpus=common_corpus, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value

print(coherence)

from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
topics = [
    ['human', 'computer', 'system', 'interface'],
    ['graph', 'minors', 'trees', 'eps']
         ]

cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value
print(coherence)

'''
https://radimrehurek.com/gensim/models/coherencemodel.html
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

Evaluating Topic Models
https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

'''
