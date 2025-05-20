from flask import Blueprint, request, redirect, url_for, flash
from ..models import db, Expense

delete_expense_bp = Blueprint('delete_expense', __name__)

@delete_expense_bp.route('/delete_expense/<int:expense_id>', methods=['POST'])
def delete_expense(expense_id):
    expense = Expense.query.get(expense_id)
    if expense:
        db.session.delete(expense)
        db.session.commit()
        flash('Expense deleted successfully!', 'success')
    else:
        flash('Expense not found!', 'error')
    return redirect(url_for('view_expenses.view_expenses'))