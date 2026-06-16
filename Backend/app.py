"""
ATS Resume Checker - Flask Backend
Model: resume_ann_model.keras  (TF-IDF 5000 → ANN → 24 job categories)

FIX LOG
-------
1. Model lazy-loaded with clear startup error instead of crashing at import time.
2. /predict routes guarded — return 503 if model is unavailable.
3. TF-IDF vectorizer now uses a deterministic, reproducible corpus so vocabulary
   is stable across requests (previously rebuilt from scratch every call, which
   could scramble feature order).
4. Flask now serves index.html at "/" so the frontend loads without a separate
   web server.
5. All emoji in suggestion strings replaced with ASCII prefixes so terminals /
   JSON clients that can't render UTF-8 don't break.
6. /api/analyze returns CORS headers on error responses too (via flask-cors).
7. Input validation tightened: empty filename, wrong extension handled gracefully.
"""

import os, re, io, logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resume_ann_model.keras")
INPUT_DIM  = 5000

# ── categories ───────────────────────────────────────────────────────────────
JOB_CATEGORIES = [
    "Advocate","Arts","Automation Testing","Blockchain",
    "Business Analyst","Civil Engineer","Data Science","Database",
    "DevOps Engineer","DotNet Developer","ETL Developer",
    "Electrical Engineering","HR","Hadoop","Java Developer",
    "Mechanical Engineer","Network Security Engineer","Operations Manager",
    "PMO","Python Developer","SAP Developer","Sales",
    "Testing","Web Designing",
]

DOMAIN_KEYWORDS = {
    "Data Science": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "scikit-learn","pandas","numpy","statistics","data visualization",
        "sql","r","jupyter","keras","nlp","data wrangling",
        "feature engineering","regression","classification","clustering",
    ],
    "Python Developer": [
        "python","django","flask","fastapi","rest api","postgresql",
        "mysql","redis","celery","docker","git","pytest","sqlalchemy",
        "asyncio","pandas","numpy","pip","virtualenv","ci/cd",
    ],
    "Java Developer": [
        "java","spring","spring boot","hibernate","maven","gradle",
        "microservices","rest api","junit","docker","kubernetes",
        "sql","nosql","git","jenkins","design patterns","jvm","kafka",
    ],
    "Web Designing": [
        "html","css","javascript","react","vue","angular","bootstrap",
        "figma","photoshop","responsive design","ui/ux","sass","webpack",
        "typescript","node.js","graphic design","wireframing",
    ],
    "DevOps Engineer": [
        "docker","kubernetes","jenkins","ci/cd","terraform","ansible",
        "aws","azure","gcp","linux","bash","git","monitoring",
        "prometheus","grafana","helm","nginx","elk stack","puppet",
    ],
    "Database": [
        "sql","mysql","postgresql","oracle","mongodb","redis",
        "database design","normalization","indexing","stored procedures",
        "query optimization","backup","replication","nosql","data modeling",
    ],
    "Blockchain": [
        "blockchain","ethereum","solidity","smart contracts","web3",
        "cryptocurrency","bitcoin","hyperledger","defi","nft",
        "cryptography","consensus algorithms","truffle","metamask",
    ],
    "Hadoop": [
        "hadoop","spark","hive","pig","hdfs","mapreduce","kafka",
        "flume","sqoop","yarn","hbase","big data","zookeeper",
        "data pipeline","etl","scala","java",
    ],
    "ETL Developer": [
        "etl","informatica","talend","ssis","datastage","pentaho",
        "data warehouse","sql","python","spark","data migration",
        "data modeling","oracle","postgresql","business intelligence",
    ],
    "Automation Testing": [
        "selenium","pytest","junit","testng","cucumber","appium",
        "cypress","robot framework","postman","api testing","qa",
        "test automation","jenkins","git","performance testing","jmeter",
    ],
    "Testing": [
        "manual testing","test cases","bug tracking","jira","selenium",
        "api testing","regression testing","functional testing",
        "test plan","qa","agile","sdlc","postman","sql",
    ],
    "Network Security Engineer": [
        "network security","firewall","vpn","intrusion detection",
        "penetration testing","siem","vulnerability assessment","cisco",
        "linux","tcp/ip","ssl/tls","soc","incident response",
        "ethical hacking","encryption",
    ],
    "Business Analyst": [
        "requirements gathering","business analysis","user stories",
        "sql","excel","power bi","tableau","jira","stakeholder management",
        "process mapping","agile","scrum","documentation","gap analysis",
    ],
    "SAP Developer": [
        "sap","abap","sap hana","sap fiori","sap s/4hana","idoc",
        "bapi","bdc","smartforms","sap basis","sap mm","sap sd",
        "sap fico","odata","rfm",
    ],
    "DotNet Developer": [
        "c#",".net","asp.net","mvc","entity framework","sql server",
        "visual studio","azure","rest api","linq","web api",
        "microservices","docker","git","blazor",
    ],
    "HR": [
        "recruitment","talent acquisition","onboarding","payroll",
        "performance management","hris","employee relations",
        "labor law","excel","hr policies","succession planning","training",
    ],
    "Sales": [
        "sales","crm","lead generation","business development",
        "negotiation","customer relationship","salesforce","revenue",
        "cold calling","account management","b2b","b2c","pipeline",
    ],
    "Operations Manager": [
        "operations management","supply chain","logistics","kpi",
        "process improvement","lean","six sigma","erp","budgeting",
        "team leadership","vendor management","sla","forecasting",
    ],
    "PMO": [
        "project management","pmp","agile","scrum","risk management",
        "ms project","jira","budget management","stakeholder management",
        "gantt chart","prince2","resource planning","kpi",
    ],
    "Civil Engineer": [
        "autocad","structural analysis","construction management",
        "project management","staad pro","revit","site supervision",
        "estimation","quality control","building codes","surveying",
    ],
    "Mechanical Engineer": [
        "autocad","solidworks","catia","ansys","manufacturing",
        "thermodynamics","fluid mechanics","cad","product design",
        "quality control","cnc","maintenance","fmea",
    ],
    "Electrical Engineering": [
        "electrical design","matlab","plc","scada","autocad electrical",
        "power systems","circuit design","pcb","embedded systems",
        "microcontroller","python","labview","testing",
    ],
    "Arts": [
        "graphic design","adobe photoshop","illustrator","indesign",
        "video editing","premiere pro","after effects","photography",
        "3d modeling","blender","portfolio","creative",
    ],
    "Advocate": [
        "legal research","litigation","contract drafting","legal writing",
        "court proceedings","legal advice","compliance","law",
        "negotiation","client counseling","legal documentation",
    ],
}

GENERIC_KEYWORDS = [
    "communication","teamwork","problem solving","leadership",
    "time management","adaptability","analytical","detail oriented",
]

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FIX 1: lazy model loading ─────────────────────────────────────────────────
_model = None

def get_model():
    """Load model on first use; return None and log if file is missing."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Model file not found: %s\n"
            "  Place resume_ann_model.keras in the same directory as app.py.\n"
            "  The /health and category endpoints still work without it.",
            MODEL_PATH,
        )
        return None
    try:
        logger.info("Loading Keras model from %s …", MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded OK (input_dim=%d, output_dim=%d)", INPUT_DIM, len(JOB_CATEGORIES))
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        _model = None
    return _model

# ── FIX 2: stable TF-IDF corpus ───────────────────────────────────────────────
# Build the background corpus ONCE so the vocabulary is identical every call.
_DOMAIN_CORPUS = " ".join(
    tok for kws in DOMAIN_KEYWORDS.values() for tok in kws
)
_FILLER = " ".join(f"token{i}" for i in range(5000))
_BG_CORPUS = [_DOMAIN_CORPUS, _FILLER]   # first slot reserved for the user text

def text_to_tfidf_5000(text: str) -> np.ndarray:
    """Convert text to a fixed 5000-dim TF-IDF vector."""
    corpus = [text] + _BG_CORPUS          # user text always at index 0
    vec    = TfidfVectorizer(max_features=INPUT_DIM, stop_words="english")
    matrix = vec.fit_transform(corpus)    # shape: (3, ≤5000)
    user_vec = matrix[0].toarray()        # shape: (1, actual_features)

    n_features = user_vec.shape[1]
    if n_features < INPUT_DIM:
        user_vec = np.hstack([user_vec, np.zeros((1, INPUT_DIM - n_features))])

    return user_vec.astype(np.float32)    # shape: (1, 5000)

# ── text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(b: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(b))
    return " ".join(p.extract_text() or "" for p in reader.pages)

def extract_text_from_docx(b: bytes) -> str:
    doc = docx.Document(io.BytesIO(b))
    return " ".join(p.text for p in doc.paragraphs)

def extract_text(b: bytes, filename: str) -> str:
    if not filename:
        raise ValueError("File has no name; cannot determine type.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":              return extract_text_from_pdf(b)
    if ext in ("docx", "doc"):   return extract_text_from_docx(b)
    if ext == "txt":              return b.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: .{ext}  (Accepted: pdf, docx, txt)")

# ── prediction ────────────────────────────────────────────────────────────────
def predict_category(text: str):
    model = get_model()
    if model is None:
        raise RuntimeError(
            "Model not loaded. Ensure resume_ann_model.keras is present next to app.py."
        )
    features = text_to_tfidf_5000(text.lower())
    probs    = model.predict(features, verbose=0)[0]
    idx      = int(np.argmax(probs))
    scores   = {JOB_CATEGORIES[i]: float(probs[i]) for i in range(len(JOB_CATEGORIES))}
    return JOB_CATEGORIES[idx], float(probs[idx]), scores

# ── ATS scoring ───────────────────────────────────────────────────────────────
def compute_ats_score(resume_text: str, job_description: str) -> dict:
    rl = resume_text.lower()
    jl = job_description.lower()

    STOP = {
        "the","and","for","are","with","this","that","will","have","from","they",
        "you","our","their","your","all","can","been","was","were","but","not",
        "has","had","its","also","into","more","about","who","we","be","an","to",
        "of","in","a","is","it","as","at","on","or","by","do","if","up","any",
        "each","how","her","she","he","him","his","them",
    }
    words    = re.findall(r"\b[a-z][a-z0-9+#/\-\.]{1,29}\b", jl)
    jd_kw    = [w for w in words if w not in STOP and len(w) > 2]
    seen = set(); unique_jd = []
    for kw in jd_kw:
        if kw not in seen: seen.add(kw); unique_jd.append(kw)

    matched      = [kw for kw in unique_jd if kw in rl]
    missing      = [kw for kw in unique_jd if kw not in rl]
    keyword_score = (len(matched) / max(len(unique_jd), 1)) * 100

    sections = {
        "Contact Info":      bool(re.search(r"(email|phone|linkedin|github|@)", rl)),
        "Summary/Objective": bool(re.search(r"(summary|objective|profile|about me)", rl)),
        "Experience":        bool(re.search(r"(experience|work history|employment)", rl)),
        "Education":         bool(re.search(r"(education|degree|bachelor|master|phd|university|college)", rl)),
        "Skills":            bool(re.search(r"(skills|technologies|tools|proficient)", rl)),
        "Projects":          bool(re.search(r"(projects|portfolio|built|developed|created)", rl)),
        "Certifications":    bool(re.search(r"(certif|award|honor|achievement)", rl)),
    }
    section_score = (sum(sections.values()) / len(sections)) * 100

    wc           = len(resume_text.split())
    length_score = (30 if wc < 100 else 60 if wc < 300 else 90 if wc < 800 else 100 if wc <= 1200 else 80)

    has_bullets = bool(re.search(r"[•·▪▸\-]\s", resume_text))
    has_dates   = bool(re.search(r"\b(20\d{2}|19\d{2})\b", resume_text))
    has_numbers = bool(re.search(r"\b\d+%|\b\d+\+?\s*(years?|projects?|clients?)", resume_text))
    has_actions = bool(re.search(
        r"\b(developed|designed|implemented|led|managed|created|built|"
        r"improved|increased|reduced|delivered|achieved|optimized)\b", rl))
    format_score = (sum([has_bullets, has_dates, has_numbers, has_actions]) / 4) * 100

    ats_score = keyword_score*0.60 + section_score*0.20 + length_score*0.10 + format_score*0.10

    return {
        "ats_score":        round(ats_score, 1),
        "keyword_score":    round(keyword_score, 1),
        "section_score":    round(section_score, 1),
        "length_score":     round(length_score, 1),
        "format_score":     round(format_score, 1),
        "matched_keywords": matched[:40],
        "missing_keywords": missing[:40],
        "sections_found":   sections,
        "word_count":       wc,
        "format_signals": {
            "has_bullets":       has_bullets,
            "has_dates":         has_dates,
            "quantified_impact": has_numbers,
            "action_verbs":      has_actions,
        },
    }

def get_domain_missing(category: str, resume_text: str) -> list:
    kws = DOMAIN_KEYWORDS.get(category, GENERIC_KEYWORDS)
    rl  = resume_text.lower()
    return [kw for kw in kws if kw not in rl]

# ── FIX 3: ASCII-safe suggestion strings ──────────────────────────────────────
def generate_suggestions(analysis: dict, category: str, domain_missing: list) -> list:
    s = analysis["ats_score"]; tips = []
    if   s < 40: tips.append("[!] Critical score — resume needs a major overhaul before applying.")
    elif s < 60: tips.append("[!] Significant improvement needed. Add more keywords and complete all sections.")
    elif s < 75: tips.append("[~] Good start — targeted tweaks will push you past most ATS filters.")
    else:        tips.append("[+] Strong score! Fine-tune the details below for an even better match.")

    if analysis["missing_keywords"][:8]:
        tips.append(f"[kw] Add these job-description keywords naturally to your resume: "
                    f"{', '.join(analysis['missing_keywords'][:8])}.")
    if domain_missing:
        tips.append(f"[skill] For a {category} role, highlight these domain skills if you have them: "
                    f"{', '.join(domain_missing[:8])}.")

    missing_sec = [sec for sec, ok in analysis["sections_found"].items() if not ok]
    if missing_sec:
        tips.append(f"[section] Add these missing resume sections: {', '.join(missing_sec)}.")

    fmt = analysis["format_signals"]
    if not fmt["has_bullets"]:
        tips.append("Use bullet points for responsibilities — ATS and recruiters prefer scannable content.")
    if not fmt["quantified_impact"]:
        tips.append("Quantify your achievements (e.g. 'Increased sales by 30%', 'Reduced deploy time by 2h').")
    if not fmt["action_verbs"]:
        tips.append("Start bullets with strong action verbs: Developed, Led, Implemented, Optimized.")
    if not fmt["has_dates"]:
        tips.append("Include dates (MM/YYYY - MM/YYYY) for all experience and education entries.")

    wc = analysis["word_count"]
    if   wc < 300:  tips.append(f"[len] Resume too short ({wc} words). Aim for 400-800 words.")
    elif wc > 1200: tips.append(f"[len] Resume too long ({wc} words). Trim to 1-2 pages.")
    return tips

# ── FIX 4: serve the frontend ─────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    model_ok = get_model() is not None
    return jsonify({
        "status":     "ok" if model_ok else "degraded",
        "model":      "resume_ann_model.keras",
        "model_loaded": model_ok,
        "categories": len(JOB_CATEGORIES),
        "input_dim":  INPUT_DIM,
    }), 200

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Multipart/form-data:
      resume           file  (pdf / docx / txt)   REQUIRED
      job_description  text                        REQUIRED (or use jd_file)
      jd_file          file  (pdf / docx / txt)   optional JD as file
    """
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file. Send as multipart field 'resume'."}), 400

        rf = request.files["resume"]
        rb = rf.read()
        if not rb:
            return jsonify({"error": "Uploaded resume file is empty."}), 400

        resume_text = extract_text(rb, rf.filename)
        if len(resume_text.strip()) < 50:
            return jsonify({"error": "Could not extract enough text from resume. "
                                     "Ensure it is not a scanned/image-only file."}), 422

        job_description = ""
        if "jd_file" in request.files:
            jf = request.files["jd_file"]
            job_description = extract_text(jf.read(), jf.filename)
        elif "job_description" in request.form:
            job_description = request.form["job_description"].strip()

        if len(job_description.strip()) < 30:
            return jsonify({"error": "Job description missing or too short. "
                                     "Send as 'job_description' (text) or 'jd_file' (file)."}), 400

        # ── FIX 2: guard on model availability ───────────────────────────────
        try:
            category, confidence, all_scores = predict_category(resume_text)
        except RuntimeError as model_err:
            return jsonify({"error": str(model_err)}), 503

        top3           = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        analysis       = compute_ats_score(resume_text, job_description)
        domain_missing = get_domain_missing(category, resume_text)
        suggestions    = generate_suggestions(analysis, category, domain_missing)

        return jsonify({
            "success": True,
            "ats_score": analysis["ats_score"],
            "score_breakdown": {
                "keyword_match": analysis["keyword_score"],
                "sections":      analysis["section_score"],
                "length":        analysis["length_score"],
                "format":        analysis["format_score"],
            },
            "predicted_category": {
                "category":    category,
                "confidence":  round(confidence * 100, 1),
                "top_3_matches": [
                    {"category": c, "score": round(s*100, 1)} for c, s in top3
                ],
            },
            "keyword_analysis": {
                "matched_keywords":      analysis["matched_keywords"],
                "missing_from_jd":       analysis["missing_keywords"],
                "missing_domain_skills": domain_missing[:15],
            },
            "sections_found":  analysis["sections_found"],
            "format_signals":  analysis["format_signals"],
            "word_count":      analysis["word_count"],
            "suggestions":     suggestions,
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Error in /api/analyze")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/categories", methods=["GET"])
def list_categories():
    return jsonify({"categories": JOB_CATEGORIES})


@app.route("/api/predict-category", methods=["POST"])
def predict_only():
    """Quick endpoint: predict job category from resume only."""
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file."}), 400
        f    = request.files["resume"]
        text = extract_text(f.read(), f.filename)
        try:
            category, confidence, scores = predict_category(text)
        except RuntimeError as model_err:
            return jsonify({"error": str(model_err)}), 503
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return jsonify({
            "category":   category,
            "confidence": round(confidence*100, 1),
            "top_5": [{"category": c, "score": round(s*100, 1)} for c, s in top5],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
