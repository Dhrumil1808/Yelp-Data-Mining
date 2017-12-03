from testDataDirectory import *
from globalTrainDataDirectory import *
from util import *
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(getTrainGlobalAllDocCSR(), map(float,getTrainGlobalAllActualRating()))

def predict_global_rating(reviewCSR):
    predicted = neigh.predict(reviewCSR)
    ans = {}
    ans["knn"] = predicted[0]
    return ans
