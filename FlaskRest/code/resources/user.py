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
        if UserModel.find_by_username(data['username']):
            UserModel(data['username'], data['password']).save_user()
            return {'message': f'username {data["username"]} created'}

        return {'message': 'user registered successfully',
                'user': data
        }, 201

