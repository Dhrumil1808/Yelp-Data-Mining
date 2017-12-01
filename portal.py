from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from util import *

app = Flask(__name__)
api = Api(app)

filename = 'pickle/'
user = load_pickle(filename+'userid.pickle')
ratings = load_pickle(filename+'rating.pickle')
business = load_pickle(filename+'business.pickle')
docs_csr = load_pickle(filename+'docs_csr.pickle')
unique_users = set(user)
unique_business = set(business)

TODOS = {
    'user': {'task': 'API to get list of all Users'},
    'business': {'task': 'API to get all Business'},
    'predictUserTestReviews': {'task': 'API to get predicted User Reviews Test Data'},
}


def handleError(msg):
    abort(404, message=msg)

parser = reqparse.RequestParser()
parser.add_argument('task')


class User(Resource):
    def get(self):
        return np.unique(unique)


class Business(Resource):
    def get(self):
        return np.unique(business)

class UserReview(Resource):
    def get(self, userId):
        handleError('No reviews available for user :' + userId)
        return TODOS[todo_id]

class BusinessReview(Resource):
    def get(self, businessId):
        handleError('No reviews available for business :' + businessId)
        return TODOS[todo_id]

class PredictUser(Resource):
    def get(self, userId):
        handleError('No reviews available for user :' + userId)
        return TODOS[todo_id]

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


if __name__ == '__main__':
    app.run(debug=True)