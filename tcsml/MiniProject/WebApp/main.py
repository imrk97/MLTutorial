# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, render_template, request

app = Flask(__name__)

user_data = {'1992309':'qwerty'}

@app.route('/')
def Home():
    return render_template('index.html')
    #return 'Hello World!'

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login_form', methods = ['POST', 'GET'])
def login_auth():
    name1 = request.form['user_id']
    pass1 = request.form['password']
    
    if name1 in user_data:
        if user_data[name1] == pass1:
            return render_template('home.html', name = name1)
        else: return 'wrong password'
    else: return 'wrong user id'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    app.run(debug = True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
