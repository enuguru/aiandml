from pathlib import Path

class Config:
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{Path(__file__).parent}/expenses.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your_secret_key'  # Change this to a random secret key for production use.