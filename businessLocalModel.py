from testDataDirectory import *
from trainDataDirectory import *
from util import *
from sklearn.neighbors import KNeighborsRegressor

def getBusinessTrainData(businessId):    
    reviewrows =[]
    reviewratings=[]
    counter=0
    for b in getTestAllBusiness():
        if(b==businessId):
            reviewrows.append(counter) 
            reviewratings.append(float(getTrainAllActualRating()[counter]))
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

def predict_business_local_rating(businessId,reviewCSR):
    reviewrows,reviewratings = getBusinessTrainData(businessId)
    review_csr_x=getTrainAllDocCSR()[reviewrows]
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(review_csr_x, reviewratings) 
    
    test_reviewrows,test_reviewratings = getBusinessTestData(businessId)
    predicted = neigh.predict(reviewCSR)
    ans = {}
    ans["knn"] = predicted[0]
    return ans
