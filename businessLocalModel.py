from trainBusinessDataDirectory import *
from testDataDirectory import *
from util import *
from sklearn.neighbors import KNeighborsRegressor
from SentimentAnalysis import *
from NMFModel import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

def getBusinessTrainData(businessId):
    reviewrows =[]
    reviewratings=[]
    counter=0
    for b in getTrainBusinessAllBusiness():
        if(b==businessId):
            reviewrows.append(counter)
            reviewratings.append(float(getTrainBusinessAllActualRating()[counter]))
        counter=counter+1

    return reviewrows,reviewratings

def getBusinessTestData(businessId):
    reviewrows =[]
    reviewratings=[]
    counter=0
    for b in getTestAllBusiness():
        if(b==businessId):
            reviewrows.append(counter)
            reviewratings.append(float(getTestAllActualRating()[counter]))
        counter=counter+1

    return reviewrows,reviewratings

def predict_business_local_rating(businessId,reviewdata):
    reviewrows,reviewratings = getBusinessTrainData(businessId)

    print len(reviewrows)

    train_reviews = getTrainBusinessAllDocFullI(reviewrows)
    predicted_sentiment = predictSentiment(train_reviews,reviewdata,reviewratings)
    train_reviews.append(reviewdata)
    reviewratings.append(predicted_sentiment)

    nmdf,vocab = nmfsentimentModel(train_reviews,reviewratings, n_components=5,n_top_words=10,n_features=1000)
    test_reviewrows,test_reviewratings = getBusinessTestData(businessId)

    ans={}

    clfs = [ LogisticRegression(),
    KNeighborsRegressor(n_neighbors=3),
    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    ]
    clf_names = ['Logistic Regression','KNeighbors Regressor','SVR']

    for (i, clf_) in enumerate(clfs):
            neigh=clf_.fit(nmdf[:-1], reviewratings[:-1])
            predicted = neigh.predict(nmdf[len(train_reviews)-1:])
            ans[clf_names[i]] = predicted[0]
    return ans
