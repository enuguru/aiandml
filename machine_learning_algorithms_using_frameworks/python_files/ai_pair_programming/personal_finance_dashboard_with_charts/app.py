from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Database setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "finance.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ✅ Define User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# ✅ Fix: Ensure user_loader function exists
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ✅ Define Dashboard Route (Previously Missing)
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")  # Ensure this template exists

# ✅ Register Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! You can log in now.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ✅ Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)  # ✅ Log the user in
            return redirect(url_for("dashboard"))  # Ensure this matches @app.route("/dashboard")
        else:
            flash("Invalid credentials. Please try again.", "danger")

    return render_template("login.html")

# ✅ Logout Route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

# ✅ Home Route
@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # ✅ Ensure database tables are created
    app.run(debug=True)
