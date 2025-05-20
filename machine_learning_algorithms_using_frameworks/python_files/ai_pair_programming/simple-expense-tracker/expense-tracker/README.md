# Expense Tracker Web Application

This is a simple Expense Tracker web application built using Python, Flask, and SQLite. The application allows users to add, view, and delete expenses, with data stored using Flask-SQLAlchemy.

## Project Structure

```
expense-tracker
├── app.py                # Entry point of the application
├── config.py             # Configuration settings for Flask
├── models.py             # Database models using Flask-SQLAlchemy
├── routes                # Contains route handlers for the application
│   ├── __init__.py       # Initializes the routes package
│   ├── add_expense.py    # Route for adding a new expense
│   ├── view_expenses.py   # Route for viewing all expenses
│   └── delete_expense.py  # Route for deleting an expense
├── templates             # HTML templates for rendering views
│   ├── base.html         # Base template for the application
│   ├── add_expense.html  # Form for adding a new expense
│   ├── view_expenses.html # Displays all expenses
│   └── delete_expense.html # Confirmation for expense deletion
├── static                # Static files such as CSS
│   └── style.css         # CSS styles for the application
├── requirements.txt      # Lists dependencies for the project
└── README.md             # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd expense-tracker
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   python app.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

- **Add Expense:** Navigate to the add expense page to submit a new expense.
- **View Expenses:** View all recorded expenses in a table format.
- **Delete Expense:** Confirm deletion of an expense to remove it from the records.

## License

This project is licensed under the MIT License.