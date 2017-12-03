from util import *


trainBusiness_filename = 'pickle/train/businessid/'
trainBusiness_user = load_pickle(trainBusiness_filename+'userid.pickle')
trainBusiness_ratings = load_pickle(trainBusiness_filename+'rating.pickle')
trainBusiness_business = load_pickle(trainBusiness_filename+'business.pickle')
trainBusiness_docs_csr = load_pickle(trainBusiness_filename+'docs_csr.pickle')
trainBusiness_unique_users = set(trainBusiness_user)
trainBusiness_unique_business = set(trainBusiness_business)

def getTrainBusinessAllUsers():
    return trainBusiness_user

def getTrainBusinessAllBusiness():
    return trainBusiness_business

def getTrainBusinessAllActualRating():
    return trainBusiness_ratings

def getTrainBusinessAllDocCSR():
    return trainBusiness_docs_csr

def getTrainBusinessUser(i):
    return trainBusiness_user[i]

def getTrainBusinessBusiness(i):
    return trainBusiness_business[i]

def getTrainBusinessActualRating(i):
    return trainBusiness_ratings[i]


def getTrainBusinessDocCSR(i):
    return trainBusiness_docs_csr[i]

def getTrainBusinessUniqueUsers():
    return trainBusiness_unique_users

def getTrainBusinessUniqueBusiness():
    return trainBusiness_unique_business

