# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import (
    Flask, 
    render_template, 
    request,
    session,
    redirect,
    url_for,
    g, 
    abort
)

app = Flask(__name__)
app.secret_key = "123i"
#user_data = {'1992309': 'qwerty'}

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User:{self.username}>'
        
users = list()
users.append(User(id=1, username="rohan", password="1234"))
users.append(User(id=2, username="saahil", password="1234"))
users.append(User(id=3, username="aamir", password="1234"))



@app.route('/')
def Home():
    return render_template('index.html')
    # return 'Hello World!'

@app.before_request
def before_request():
    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        session.pop('user_id', None)
        username = request.form['username']
        password = request.form['password']

        user = [x for x in users if x.username == username][0]

        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/home')
def home():
    if not g.user:
        return redirect(url_for('login'))
    return render_template('home.html')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
