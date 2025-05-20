from flask import Blueprint, render_template, request, redirect, url_for, flash
from ..models import db, Expense

add_expense_bp = Blueprint('add_expense', __name__)

@add_expense_bp.route('/add', methods=['GET', 'POST'])
def add_expense():
    if request.method == 'POST':
        description = request.form.get('description')
        amount = request.form.get('amount')
        category = request.form.get('category')

        if not description or not amount or not category:
            flash('All fields are required!', 'error')
            return redirect(url_for('add_expense.add_expense'))

        new_expense = Expense(description=description, amount=float(amount), category=category)
        db.session.add(new_expense)
        db.session.commit()
        flash('Expense added successfully!', 'success')
        return redirect(url_for('view_expenses.view_expenses'))

    return render_template('add_expense.html')