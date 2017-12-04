from testDataDirectory import *
from trainDataDirectory import *
from util import *
from sklearn.neighbors import KNeighborsRegressor
from NMFModel import *
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from SentimentAnalysis import *

def getUserTrainData(userid):
    reviewrows =[]
    reviewratings=[]
    counter=0
    for u in getTrainAllUsers():
        if(u==userid):
            reviewrows.append(counter)
            reviewratings.append(float(getTrainAllActualRating()[counter]))
        counter=counter+1

    return reviewrows,reviewratings

def getUserTestData(userid):
    reviewrows =[]
    reviewratings=[]
    counter=0
    for u in getTestAllUsers():
        if(u==userid):
            reviewrows.append(counter)
            reviewratings.append(float(getTestAllActualRating()[counter]))
        counter=counter+1

    return reviewrows,reviewratings

def predict_user_local_rating(userid,reviewdata):
    reviewrows,reviewratings = getUserTrainData(userid)

    print len(reviewrows)

    train_reviews = getTrainAllDocFullI(reviewrows)
    print len(train_reviews)

    predicted_sentiment = predictSentiment(train_reviews,reviewdata,reviewratings)
    train_reviews.append(reviewdata)

    reviewratings.append(predicted_sentiment)

    nmdf,vocab = nmfsentimentModel(train_reviews,reviewratings, n_components=5,n_top_words=10,n_features=1000)
    test_reviewrows,test_reviewratings = getUserTestData(userid)

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

#ans =predict_user_local_rating("C8W0VzsXaTg4YvciNTy3bg","I agree with most of the reviews. Great food, really cheap, absolutely worth it.  Seems like there is very little or no msg on their food. On the downside, msg does enhance the flavor so I did find the everything a tad blander than in a typical hakka restaurant. On the upside, there is no msg! I did not feel sick after the meal. As stated in several posts, what is served here seems healthier.");

#print ans
