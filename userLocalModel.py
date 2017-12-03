from testDataDirectory import *
from trainDataDirectory import *
from util import *
from sklearn.neighbors import KNeighborsRegressor

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

def predict_user_local_rating(userid,reviewCSR):
    reviewrows,reviewratings = getUserTrainData(userid)
    review_csr_x=getTrainAllDocCSR()[reviewrows]
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(review_csr_x, reviewratings) 
    
    test_reviewrows,test_reviewratings = getUserTestData(userid)
    predicted = neigh.predict(reviewCSR)
    ans = {}
    ans["knn"] = predicted[0]
    return ans
