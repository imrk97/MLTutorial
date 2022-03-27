import sqlite3
from flask_restful import Resource, reqparse
from models.user import UserModel
class UserRegister(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('username',
                        type=str,
                        required=True,
                        help='This field cannot be left blank.'
                        )
    parser.add_argument('password',
                        type=str,
                        required=True,
                        help='This field cannot be left blank.'
                        )
    @classmethod
    def post(cls):
        data = cls.parser.parse_args()
        user = UserModel.find_by_username(data['username'])
        print(user)
        if user is not None :
            
            return {'message': f'username {data["username"]} is already there.'}


        user = UserModel(data['username'], data['password'])
        print(user)
        user.save_user()
        return {'message': f'username {data["username"]} created'}

