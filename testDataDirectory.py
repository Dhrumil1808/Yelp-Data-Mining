from util import *


test_filename = 'pickle/'
test_user = load_pickle(test_filename+'userid.pickle')
test_ratings = load_pickle(test_filename+'rating.pickle')
test_business = load_pickle(test_filename+'business.pickle')
test_docs_csr = load_pickle(test_filename+'docs_csr.pickle')
test_unique_users = set(test_user)
test_unique_business = set(test_business)

def getTestAllUsers():
    return test_user

def getTestAllBusiness():
    return test_business

def getTestAllActualRating():
    return test_ratings

def getTestAllDocCSR():
    return test_docs_csr

def getTestUser(i):
    return test_user[i]

def getTestBusiness(i):
    return test_business[i]

def getTestActualRating(i):
    return test_ratings[i]

def getTestDocCSR(i):
    return test_docs_csr[i]

def getTestUniqueUsers():
    return test_unique_users

def getTestUniqueBusiness():
    return test_unique_business

