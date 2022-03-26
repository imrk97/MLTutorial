from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

stores = [
	{
		'name': 'MyStore',
		'items': [
			{
				'name': 'Mystore_Item1',
				'price': 15.99
			},
			{
				'name': 'Mystore_Item2',
				'price': 9.99
			}
		]
	}
]


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/store', methods=['POST'])
def create_store():
	request_data = request.get_json()
	print(request_data)
	new_store = {
		'name': request_data['name'],
		'items': []
	}
	stores.append(new_store)
	return jsonify(new_store)


@app.route('/store/<string:name>')
def get_store(name):
	for store in stores:
		if store['name'] == name:
			return jsonify(store)
	return jsonify({'message': 'ERROR: Store name not found.'})

@app.route('/store')
def get_stores():
	return jsonify({'stores':stores})

@app.route('/store/<string:name>/item', methods=['POST'])
def create_item_in_store(name):
	request_data = request.get_json()
	for store in stores:
		if store['name'] == name:
			new_item = {
				'name': request_data['name'],
				'price': request_data['price']
			}
			store['items'].append(new_item)
			return jsonify(new_item)
	return jsonify({'message': 'ERROR: Store name not found.'})

@app.route('/store/<string:name>/item')
def get_items_in_store(name):
	for store in stores:
		if store['name'] == name:
			return jsonify({'ans': store['items']})
	return jsonify({'message': 'ERROR: Store name not found.'})

app.run(port=5000)