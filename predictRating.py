from testDataDirectory import *
from trainDataDirectory import *
from userLocalModel import *
from businessLocalModel import *
from globalModel import *

def predictRating(userid, businessid, reviewCSR):
    predicted = predict_user_local_rating(userid,reviewCSR)
    ans = {}
    ans["userLocalModelPrediction"] = predicted
    p2 = predict_business_local_rating(businessid, reviewCSR)
    ans["businessLocalModelPrediction"] = p2
    p3 = predict_global_rating(reviewCSR)
    ans["globalModelPrediction"] = p3
    ans["alpha"] = 0.5
    ans["beta"] = 0.3
    ans["gamma"] = 0.2
    ans["Final"] = {}
    ans["Final"]["knn"] = (ans["alpha"] * ans["userLocalModelPrediction"]["knn"]) + (ans["beta"] * ans["businessLocalModelPrediction"]["knn"]) + (ans["gamma"] * ans["globalModelPrediction"]["knn"] )
    return ans