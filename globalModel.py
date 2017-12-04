from testDataDirectory import *
from globalTrainDataDirectory import *
from util import *
from sklearn.neighbors import KNeighborsRegressor
from globalTrainDataDirectory import *
from SentimentAnalysis import *
from NMFModel import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

clfs = [ LogisticRegression(),
KNeighborsRegressor(n_neighbors=3),
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
]
clf_names = ['Logistic Regression','KNeighbors Regressor','SVR']
neigh=[]

train_ratings= map(float,getTrainGlobalAllActualRating())

for (i, clf_) in enumerate(clfs):
        neigh.append(clf_.fit(getTrainGlobalAllDocNMF(), train_ratings))

#neigh = KNeighborsRegressor(n_neighbors=3)
#neigh.fit(getTrainGlobalAllDocNMF(), map(float,getTrainGlobalAllActualRating()))

def predict_global_rating(reviewCSR):
    reviewCSR= [reviewCSR]
    ans = {}

    for (i, clf_) in enumerate(clfs):

            predicted = neigh[i].predict(reviewCSR)
            ans[clf_names[i]] = predicted[0]
    return ans
    #return ans
