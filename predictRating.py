from testDataDirectory import *
from trainDataDirectory import *
from userLocalModel import *
from businessLocalModel import *
from globalModel import *

def predictRating(userid, businessid, reviewCSR,reviewNMF):
    predicted = predict_user_local_rating(userid,reviewCSR)
    ans = {}
    ans["userLocalModelPrediction"] = predicted
    p2 = predict_business_local_rating(businessid, reviewCSR)
    ans["businessLocalModelPrediction"] = p2
    p3 = predict_global_rating(reviewNMF)
    ans["globalModelPrediction"] = p3
    ans["alpha"] = 0.5
    ans["beta"] = 0.3
    ans["gamma"] = 0.2
    ans["Final"] = {}
    ans["Final"]["KNeighbors Regressor"] = (ans["alpha"] * ans["userLocalModelPrediction"]["KNeighbors Regressor"]) + (ans["beta"] * ans["businessLocalModelPrediction"]["KNeighbors Regressor"]) + (ans["gamma"] * ans["globalModelPrediction"]["KNeighbors Regressor"] )
    ans["Final"]["Logistic Regression"] = (ans["alpha"] * ans["userLocalModelPrediction"]["Logistic Regression"]) + (ans["beta"] * ans["businessLocalModelPrediction"]["Logistic Regression"]) + (ans["gamma"] * ans["globalModelPrediction"]["Logistic Regression"] )
    ans["Final"]["SVR"] = (ans["alpha"] * ans["userLocalModelPrediction"]["SVR"]) + (ans["beta"] * ans["businessLocalModelPrediction"]["SVR"]) + (ans["gamma"] * ans["globalModelPrediction"]["SVR"] )
    return ans
