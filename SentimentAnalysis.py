from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB

def transformRating(rating):
    transformed=list()
    for r in rating:
        if(r<3.5):
            transformed.append(1)
        else:
            transformed.append(5)
    return transformed


#transformRatings=transformRating(rating)


def predictSentiment(train_reviews,review,ratings):
    n_features = 1000

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
    tratings = transformRating(ratings)
    print train_reviews
    sentimentXtrain = tfidf_vectorizer.fit_transform(train_reviews)
    lst = []
    lst.append(review)
    sentimentXtest = tfidf_vectorizer.transform(lst)
    #sentimentYtrain = transformRating

    classifier = MultinomialNB()
    clf = classifier.fit(sentimentXtrain, tratings)

    preds = clf.predict(sentimentXtest)
    return preds
