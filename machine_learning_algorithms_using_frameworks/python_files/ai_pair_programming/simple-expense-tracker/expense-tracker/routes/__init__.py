from flask import Blueprint

# Create a blueprint for the routes
expense_tracker_bp = Blueprint('expense_tracker', __name__)

# Import the route modules to register them with the blueprint
from .add_expense import *
from .view_expenses import *
from .delete_expense import *