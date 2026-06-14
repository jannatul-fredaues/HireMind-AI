# ATS Resume Checker — Backend

Flask REST API powered by `resume_ann_model.keras` (ANN with 24 job categories).

## Setup

```bash
pip install -r requirements.txt
```

Place `resume_ann_model.keras` in the same folder as `app.py`.

## Run

```bash
# Development
python app.py

# Production
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## API Reference

### `GET /health`
Returns server status.

**Response:**
```json
{ "status": "ok", "model": "resume_ann_model.keras", "categories": 24, "input_dim": 5000 }
```

---

### `POST /api/analyze` ← Main endpoint
Accepts `multipart/form-data`.

| Field | Type | Required | Description |
|---|---|---|---|
| `resume` | file | ✅ | PDF, DOCX, or TXT |
| `job_description` | text | ✅ | Plain text JD |
| `jd_file` | file | optional | JD as file instead of text |

**Response:**
```json
{
  "success": true,
  "ats_score": 78.4,
  "score_breakdown": {
    "keyword_match": 68.0,
    "sections": 85.7,
    "length": 90.0,
    "format": 100.0
  },
  "predicted_category": {
    "category": "Python Developer",
    "confidence": 72.3,
    "top_3_matches": [...]
  },
  "keyword_analysis": {
    "matched_keywords": ["python", "docker", "flask", ...],
    "missing_from_jd": ["kubernetes", "fastapi", ...],
    "missing_domain_skills": ["asyncio", "celery", ...]
  },
  "sections_found": {
    "Contact Info": true,
    "Summary/Objective": true,
    "Experience": true,
    "Education": true,
    "Skills": true,
    "Projects": false,
    "Certifications": false
  },
  "format_signals": {
    "has_bullets": true,
    "has_dates": true,
    "quantified_impact": true,
    "action_verbs": true
  },
  "word_count": 512,
  "suggestions": [
    "✅ Strong score! Fine-tune the details below...",
    "📌 Add these job-description keywords...",
    "📄 Add these missing resume sections: Projects, Certifications."
  ]
}
```

---

### `POST /api/predict-category`
Predict job category from resume only (no JD needed).

| Field | Type | Required |
|---|---|---|
| `resume` | file | ✅ |

**Response:**
```json
{
  "category": "Data Science",
  "confidence": 84.2,
  "top_5": [{"category": "Data Science", "score": 84.2}, ...]
}
```

---

### `GET /api/categories`
Returns all 24 supported job categories.

---

## ATS Score Breakdown

| Component | Weight | What it measures |
|---|---|---|
| Keyword Match | 60% | JD keywords present in resume |
| Sections | 20% | Contact, Summary, Experience, Education, Skills, Projects, Certs |
| Length | 10% | Optimal: 400–1200 words |
| Format | 10% | Bullets, dates, numbers, action verbs |

## Frontend Integration (JavaScript)

```javascript
const formData = new FormData();
formData.append("resume", resumeFile);
formData.append("job_description", jobDescriptionText);

const response = await fetch("http://localhost:5000/api/analyze", {
  method: "POST",
  body: formData,
});
const result = await response.json();
console.log("ATS Score:", result.ats_score);
console.log("Suggestions:", result.suggestions);
```

## Supported Job Categories (24)

Advocate, Arts, Automation Testing, Blockchain, Business Analyst,
Civil Engineer, Data Science, Database, DevOps Engineer, DotNet Developer,
ETL Developer, Electrical Engineering, HR, Hadoop, Java Developer,
Mechanical Engineer, Network Security Engineer, Operations Manager,
PMO, Python Developer, SAP Developer, Sales, Testing, Web Designing
