from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

# Database configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "expense.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Expense Model
class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

@app.route("/")
def home():
    expenses = Expense.query.all()
    return render_template("index.html", expenses=expenses)

@app.route("/add", methods=["POST"])
def add_expense():
    description = request.form.get("description")
    amount = request.form.get("amount")
    category = request.form.get("category")

    if description and amount and category:
        new_expense = Expense(description=description, amount=float(amount), category=category)
        db.session.add(new_expense)
        db.session.commit()

    return redirect(url_for("home"))

@app.route("/delete/<int:expense_id>")
def delete_expense(expense_id):
    expense = Expense.query.get(expense_id)
    if expense:
        db.session.delete(expense)
        db.session.commit()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
