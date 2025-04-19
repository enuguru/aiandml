from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import PyPDF2
import spacy
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///resumes.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"

db = SQLAlchemy(app)

# Load NLP model
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

def parse_resume(text):
    doc = nlp(text)
    
    # Extract Name
    name = "Unknown"
    if doc.ents:
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

    return name, email, phone, ", ".join(set(skills)), text

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
    name, email, phone, skills, parsed_text = parse_resume(text)

    new_resume = Resume(name=name, email=email, phone=phone, skills=skills, text=parsed_text)
    db.session.add(new_resume)
    db.session.commit()

    return jsonify({"name": name, "email": email, "phone": phone, "skills": skills})

@app.route("/match", methods=["POST"])
def match_resume():
    job_description = request.json["job_description"]
    resumes = Resume.query.all()

    # Prepare Text Data
    resume_texts = [r.text for r in resumes]
    resume_texts.append(job_description)

    # Compute Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(resume_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Rank Candidates
    results = []
    for i, score in enumerate(cosine_similarities[0]):
        results.append({"name": resumes[i].name, "email": resumes[i].email, "match_score": round(score * 100, 2)})

    results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
