from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import PyPDF2
import spacy
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///resumes.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"

db = SQLAlchemy(app)
nlp = spacy.load("en_core_web_sm")

# Create Upload Directory if not Exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Resume Model
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(50))
    skills = db.Column(db.Text)
    experience = db.Column(db.Integer)  # Years of experience
    text = db.Column(db.Text)

with app.app_context():
    db.create_all()

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_experience(text):
    experience_years = re.findall(r"(\d+)\s+years", text.lower())
    return max(map(int, experience_years)) if experience_years else 0  # Return max years found

def parse_resume(text):
    doc = nlp(text)

    # Extract Name
    name = "Unknown"
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Extract Email & Phone
    email = None
    phone = None
    for token in doc:
        if "@" in token.text:
            email = token.text
        if token.like_num and len(token.text) >= 10:
            phone = token.text

    # Extract Skills
    skills = []
    skills_list = ["Python", "Flask", "Machine Learning", "Data Science", "Java", "SQL", "AI", "Deep Learning"]
    for token in doc:
        if token.text in skills_list:
            skills.append(token.text)

    # Extract Experience
    experience = extract_experience(text)

    return name, email, phone, ", ".join(set(skills)), experience, text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract and Parse Resume
    text = extract_text_from_pdf(filepath)
    name, email, phone, skills, experience, parsed_text = parse_resume(text)

    new_resume = Resume(name=name, email=email, phone=phone, skills=skills, experience=experience, text=parsed_text)
    db.session.add(new_resume)
    db.session.commit()

    return jsonify({"name": name, "email": email, "phone": phone, "skills": skills, "experience": experience})

@app.route("/match", methods=["POST"])
def match_resume():
    job_description = request.json["job_description"].lower()
    resumes = Resume.query.all()

    # Extract job-related keywords
    job_skills = set(job_description.split())  # Simple word split (improve with NLP)

    results = []
    for resume in resumes:
        resume_skills = set(resume.skills.lower().split(", "))  # Convert skills to set
        skill_match_score = len(resume_skills & job_skills) / len(job_skills) if job_skills else 0

        # Compute similarity using NLP for better ranking
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume.text, job_description])
        text_match_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        # Weighted scoring: 70% text match + 30% skill match + experience bonus
        final_score = (text_match_score * 70) + (skill_match_score * 30) + (resume.experience * 2)

        results.append({
            "name": resume.name,
            "email": resume.email,
            "experience": resume.experience,
            "match_score": round(final_score, 2)
        })

    # Sort by highest match score
    results = sorted(results, key=lambda x: x["match_score"], reverse=True)

    return jsonify(results)
if __name__ == "__main__":
    app.run(debug=True)
