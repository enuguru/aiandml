from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

from routes import add_expense, view_expenses, delete_expense

if __name__ == '__main__':
    app.run(debug=True)