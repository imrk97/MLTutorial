import sqlite3
from flask_restful import Resource, reqparse
from flask_jwt import jwt_required
from models.item import ItemModel
from models.store import StoreModel


class Store(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument(
        "name", type=float, required=True, help="This field cannot be left blank."
    )

    def get(self, name):
        store = StoreModel.find_by_name(name)
        if store:
            return store.json()
        return {"message": f"Store {name} not found"}, 404

    def post(self, name):
        store = StoreModel.find_by_name(name)
        if store:
            return {"message": f"store {name} already exists."}
        store = StoreModel(name)
        store.save_to_db()
        return store.json()

    def delete(self, name):
        store = StoreModel.find_by_name(name)
        if store:
            store.delete_from_db()
            return {"message": f"Store {name} deleted"}
        return {"message": f"store {name} does not exist."}


class StoreList(Resource):
    def get(self):
        return {"stores": [store.json() for store in StoreModel.get_all()]}
