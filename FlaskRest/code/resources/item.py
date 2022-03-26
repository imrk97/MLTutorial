import sqlite3
from flask_restful import Resource, reqparse
from flask_jwt import jwt_required
from models.item import ItemModel


class Item(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument(
        "price", type=float, required=True, help="This field cannot be left blank."
    )

    @jwt_required()
    # @classmethod
    def get(cls, name):

        item = ItemModel.find_by_name(name)
        if item:
            return item.json()
        return {"message": f"item {name} is not in the db"}

    @classmethod
    def post(cls, name):
        data = cls.parser.parse_args()
        item = ItemModel(name, data["price"])
        item.save_to_db()
        return item.json()

    @classmethod
    def put(cls, name):
        data = cls.parser.parse_args()
        item = ItemModel.find_by_name(name)
        if item:
            item.price = data["price"]
            item.save_to_db()
            return item.json()
        return {"message": f"There is no item named {name}"}

    @classmethod
    def delete(cls, name):
        # data = cls.parser.parse_args()
        item = ItemModel.find_by_name(name)
        if item:
            item.delete_from_db()
            return {"message": f"Item {name} deleted"}
        return {"message": f"Item {name} not found in the database"}


class ItemList(Resource):
    def get(self):
        items = ItemModel.get_all()
        if len(items) == 0:
            return {"message": "No Items in the table"}
        items_str = []
        for item in items:
            items_str.append(item.json())
        return {"items": items_str}
