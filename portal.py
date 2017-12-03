from flask import Flask, json
from flask_restful import reqparse, abort, Api, Resource
from util import *
import random
from predictRating import *
from testDataDirectory import *

app = Flask(__name__)
api = Api(app)

# test_filename = 'pickle/'
# test_user = load_pickle(test_filename+'userid.pickle')
# test_ratings = load_pickle(test_filename+'rating.pickle')
# test_business = load_pickle(test_filename+'business.pickle')
# test_docs_csr = load_pickle(test_filename+'docs_csr.pickle')
# test_unique_users = set(test_user)
# test_unique_business = set(test_business)

TODOS = {
    'user': {'task': 'API to get list of all Users'},
    'business': {'task': 'API to get all Business'},
    'predictUserTestReviews': {'task': 'API to get predicted User Reviews Test Data'},
}


def handleError(msg):
    abort(404, message=msg)

parser = reqparse.RequestParser()
parser.add_argument('task')

class TDObject:
    def __init__(self, userId, actualRating, businessId, review ):
        self.userId = userId
        self.businessId = businessId
        self.actualRating = actualRating
        self.review = review

class TestData(Resource):
    def get(self):
        i = random.randint(0, len(test_user)-1)
        #testData = TDObject(test_user[i], test_ratings[i], test_business[i], test_docs_csr[i].todense())
        testData = {}
        testData["userId"] = getTestUser(i)
        testData["actualRating"] = getTestActualRating(i)
        testData["businessId"] = getTestBusiness(i)
        testData["reviewFull"]= getTestDoc(i)
        testData["testDataSampleId"]= i
        return testData

class PredictTestData(Resource):
    def get(self, testDataId):
        userId = test_user[int(testDataId)]
        actualRating = test_ratings[int(testDataId)]
        businessId = test_business[int(testDataId)]
        reviewCSR = test_docs_csr[int(testDataId)]
        p = predictRating(userId, businessId, reviewCSR)
        predictedResponse = {}
        predictedResponse["userId"] = userId
        predictedResponse["actualRating"] = actualRating 
        predictedResponse["businessId"] = businessId
        predictedResponse["reviewFull"] = test_docs[int(testDataId)]
        predictedResponse["predictedRating"] = p
        return predictedResponse

# Todo
# shows a single todo item and lets you delete a todo item
class Todo(Resource):
    def get(self, id):
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, id):
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    def put(self, id):
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[todo_id] = task
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        return TODOS

    # def post(self):
    #     args = parser.parse_args()
    #     todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
    #     todo_id = 'todo%i' % todo_id
    #     TODOS[todo_id] = {'task': args['task']}
    #     return TODOS[todo_id], 201

##
## Actually setup the Api resource routing here
##
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<id>')
api.add_resource(TestData, '/testData')
api.add_resource(PredictTestData, '/predictTestData/<testDataId>')

if __name__ == '__main__':
    app.run(debug=True)