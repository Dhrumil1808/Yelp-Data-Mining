from util import *


trainGlobal_filename = 'pickle/'
trainGlobal_user = load_pickle(trainGlobal_filename+'userid.pickle')
trainGlobal_ratings = load_pickle(trainGlobal_filename+'rating.pickle')
trainGlobal_business = load_pickle(trainGlobal_filename+'business.pickle')
trainGlobal_docs_csr = load_pickle(trainGlobal_filename+'docs_csr.pickle')
trainGlobal_unique_users = set(trainGlobal_user)
trainGlobal_unique_business = set(trainGlobal_business)

def getTrainGlobalAllUsers():
    return trainGlobal_user

def getTrainGlobalAllBusiness():
    return trainGlobal_business

def getTrainGlobalAllActualRating():
    return trainGlobal_ratings

def getTrainGlobalAllDocCSR():
    return trainGlobal_docs_csr

def getTrainGlobalUser(i):
    return trainGlobal_user[i]

def getTrainGlobalBusiness(i):
    return trainGlobal_business[i]

def getTrainGlobalActualRating(i):
    return trainGlobal_ratings[i]


def getTrainGlobalDocCSR(i):
    return trainGlobal_docs_csr[i]

def getTrainGlobalUniqueUsers():
    return trainGlobal_unique_users

def getTrainGlobalUniqueBusiness():
    return trainGlobal_unique_business

