from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_jwt import JWT, jwt_required
from resources.user import UserRegister
from resources.store import Store, StoreList
from datetime import timedelta

app = Flask(__name__)
from db import db
from security import authenticate, identity
from resources.item import Item, ItemList

db.init_app(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
api = Api(app)


@app.before_first_request
def create_tables():
    db.create_all()


app.secret_key = "rohan"
from security import authenticate, identity
from resources.item import Item, ItemList

app.config["JWT_EXPIRATION_DELTA"] = timedelta(seconds=1800)

jwt = JWT(app, authenticate, identity)


api.add_resource(Item, "/item/<string:name>")
api.add_resource(ItemList, "/items")
api.add_resource(UserRegister, "/register")
api.add_resource(Store, "/store/<string:name>")
api.add_resource(StoreList, "/stores")
app.run(port=5000)
