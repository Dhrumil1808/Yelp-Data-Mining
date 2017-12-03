from util import *


train_filename = 'pickle/train/uid/'
train_user = load_pickle(train_filename+'userid.pickle')
train_ratings = load_pickle(train_filename+'rating.pickle')
train_business = load_pickle(train_filename+'business.pickle')
train_docs_csr = load_pickle(train_filename+'docs_csr.pickle')
train_unique_users = set(train_user)
train_unique_business = set(train_business)

def getTrainAllUsers():
    return train_user

def getTrainAllBusiness():
    return train_business

def getTrainAllActualRating():
    return train_ratings

def getTrainAllDocCSR():
    return train_docs_csr

def getTrainUser(i):
    return train_user[i]

def getTrainBusiness(i):
    return train_business[i]

def getTrainActualRating(i):
    return train_ratings[i]


def getTrainDocCSR(i):
    return train_docs_csr[i]

def getTrainUniqueUsers():
    return train_unique_users

def getTrainUniqueBusiness():
    return train_unique_business

