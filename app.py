import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    "python", "java", "c++", "javascript", "sql", "html", "css", "react",
    "node.js", "django", "flask", "streamlit", "git", "github", "docker",
    "aws", "linux", "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "mongodb", "mysql", "rest api", "data structures",
    "algorithms", "excel", "power bi", "tableau", "statistics",
]


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join((page.extract_text() or "") for page in reader.pages)


def find_skills(text):
    text = text.lower()
    found = set()
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.add(skill)
    return found


def text_similarity(a, b):
    vectors = TfidfVectorizer(stop_words="english").fit_transform([a, b])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


st.set_page_config(page_title="AI Resume-Job Matcher", page_icon="📄")
st.title("📄 AI Resume–Job Matcher")
st.write("Upload your resume and paste a job description to see how well they match.")

resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
job_text = st.text_area("Paste job description", height=200)

if st.button("Analyze"):
    if resume_file is None or not job_text.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        resume_text = extract_text(resume_file)
        if not resume_text.strip():
            st.error("Could not read text from this PDF. Try a text-based PDF.")
        else:
            resume_skills = find_skills(resume_text)
            job_skills = find_skills(job_text)
            matched = sorted(resume_skills & job_skills)
            missing = sorted(job_skills - resume_skills)

            coverage = len(matched) / len(job_skills) if job_skills else 0
            similarity = text_similarity(resume_text, job_text)
            score = round((0.6 * coverage + 0.4 * similarity) * 100, 1)

            st.subheader(f"Match Score: {score}%")
            st.progress(min(int(score), 100))

            st.markdown("### ✅ Matched skills")
            st.write(", ".join(matched) if matched else "None found")

            st.markdown("### ❌ Missing skills")
            st.write(", ".join(missing) if missing else "None, great match!")
