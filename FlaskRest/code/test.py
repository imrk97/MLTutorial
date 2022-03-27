import sqlite3

connection = sqlite3.connect('data.db')

cursor = connection.cursor()

select_query = 'SELECT * FROM users'

for row in cursor.execute(select_query).fetchall():
        print(row)

connection.commit()
connection.close()
