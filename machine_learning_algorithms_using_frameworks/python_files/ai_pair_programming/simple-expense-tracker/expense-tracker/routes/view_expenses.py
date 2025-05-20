from flask import Blueprint, render_template
from models import Expense

view_expenses_bp = Blueprint('view_expenses', __name__)

@view_expenses_bp.route('/expenses')
def view_expenses():
    expenses = Expense.query.all()
    return render_template('view_expenses.html', expenses=expenses)