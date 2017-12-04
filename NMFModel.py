
from util import *
import time
import datetime

import cPickle as pickle
import nltk
import numpy as np
import pandas as pd
import re
import scipy as sp
import sklearn.feature_extraction.text as text
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import decomposition

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# In[2]:

def getSentiment(x):
    if x < 3.0:
        return 0
    else:
        return 1

def nmfsentimentModel(data_samples,rating, n_components=5,n_top_words=10,n_features=1000):
    n_samples = len(data_samples)
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
                                   max_features=n_features,
                                   stop_words='english')
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
    "tf-idf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
    t0 = time()

    nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)

    tfidf_nmf = nmf.transform(tfidf)

    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    vocab = np.array(tfidf_vectorizer.get_feature_names())


    #Adding sentiment layer
    trainTopics= tfidf_nmf
    trainTopics = trainTopics / np.sum(trainTopics, axis=1, keepdims=True)
    d, f = trainTopics.shape
    cols = ["Topic"+str(i) for i in xrange(1, f+1)]
    nmfDF = pd.DataFrame(trainTopics, columns=cols)

    nmfDF['rating'] = map(float,rating[:n_samples])
    nmfDF['Sentiment'] = nmfDF['rating'].map(getSentiment)
    nmfDF = nmfDF.dropna(how='any')
    return nmfDF, vocab




def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)
