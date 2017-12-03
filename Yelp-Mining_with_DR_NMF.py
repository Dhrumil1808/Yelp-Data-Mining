
# coding: utf-8

# In[1]:


from util import *
import time
import datetime

import cPickle as pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab
import re
import scipy as sp
import seaborn
import sklearn.feature_extraction.text as text

from gensim import corpora, models
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import decomposition


# In[2]:


lines=[]
with open("data/reviews_full.dat", "r") as fh:
    lines = fh.readlines()


# In[3]:


userid = []
rating = []
docs = []
business = []
i = 0
j = 0
error_line_num = []
error_lines = []
for line in lines:
    try:
        i = i + 1
        l = line.split('\t', 4)
        userid.append(l[0])
        business.append(l[1])
        rating.append(l[2])
        docs.append(l[3])
        #d = clean(l[3])
        #kmers = getKmers(d)
        #d.extend(kmers)
        #docs.append(d)
    except Exception as e:
        j = j + 1
        error_line_num.append(i)
        error_lines.append(line)

print 'Training Data: Number of lines processed: ' + str(i)
print 'Training Data: Length of userid array: ' + str(len(userid))
print 'Training Data: Length of rating array: ' + str(len(rating))
print 'Training Data: Length of docs array: ' + str(len(docs))
print 'Training Data: Length of business array: ' + str(len(business))
print 'Training Data: Number of exceptions encountered: ' + str(j)



# In[ ]:



from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
n_samples = 50000
n_features = 15000
n_components = 20
n_top_words = 50

data_samples = docs[:n_samples]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
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
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
vocab = np.array(tfidf_vectorizer.get_feature_names())
#print_top_words(nmf, tfidf_feature_names, n_top_words)


def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)

save_pickle(tfidf_nmf,"nmf.pickle")
save_pickle(vocab,"vocab.pickle")


# In[6]:


trainTopics= tfidf_nmf
trainTopics = trainTopics / np.sum(trainTopics, axis=1, keepdims=True)


# In[7]:


print(trainTopics)


# In[8]:


d, f = trainTopics.shape
cols = ["Topic"+str(i) for i in xrange(1, f+1)]
nmfDF = pd.DataFrame(trainTopics, columns=cols)


# In[9]:


nmfDF['rating'] = map(float,rating[:50000])


# In[10]:


nmfDF.T.head(25)


# In[11]:


def getSentiment(x):
    if x < 3.5:
        return 0
    else:
        return 1


# In[12]:


nmfDF['Sentiment'] = nmfDF['rating'].map(getSentiment)


# In[13]:


nmfDF = nmfDF.dropna(how='any')


# In[14]:


nmfDF.T.head(25)


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsRegressor


# In[16]:


cols = [u'Topic1', u'Topic2', u'Topic3', u'Topic4', u'Topic5', u'Topic6', u'Topic7', u'Topic8', u'Topic9', u'Topic10', u'Topic11', u'Topic12', u'Topic13', u'Topic14', u'Topic15', u'Topic16', u'Topic17', u'Topic18', u'Topic19', u'Topic20', u'Sentiment']
Xtrain = nmfDF[:40000][cols]
Ytrain = nmfDF[:40000]['rating']
Xtest = nmfDF[40000:][cols]
Ytest = nmfDF[40000:]['rating']


#Xtrain_base = trainVectors[:18000]
#Ytrain_base = map(float,rating[:18000])
#Xtest_base = trainVectors[8000:10000]
#Ytest_base = map(float,rating[8000:10000])




# In[17]:


from sklearn.metrics import mean_squared_error, r2_score
clfs = [ LogisticRegression(),KNeighborsRegressor(n_neighbors=3)]
clf_names = ['Logistic Regression','KNeighborsRegressor']
results = {}
for (i, clf_) in enumerate(clfs):
    clf = clf_.fit(Xtrain, Ytrain)
    preds = clf.predict(Xtest)

    #clf1 = clf_.fit(Xtrain_base, Ytrain_base)
    #preds1 = clf.predict(Xtest_base)



    #precision = metrics.precision_score(Ytest, preds)
    #recall = metrics.recall_score(Ytest, preds)
    #f1 = metrics.f1_score(Ytest, preds)
    #accuracy = accuracy_score(Ytest, preds)
    #report = classification_report(Ytest, preds)
    #matrix = metrics.confusion_matrix(Ytest, preds, labels=[1, 2, 3, 4, 5])
    score= r2_score(Ytest, preds)
    #score1= r2_score(Ytest_base, preds1)
    print "NMF: " +clf_names[i] + str(score)
    #print "Baseline : " +clf_names[i] + str(score)



# In[ ]:


print results
