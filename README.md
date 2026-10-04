# 📄 AI Resume–Job Matcher

A web app that compares a resume (PDF) with a job description and shows a match score,
matched skills, and missing skills.

## How it works
1. Extracts text from the resume PDF (pypdf)
2. Finds known tech skills in both texts
3. Computes text similarity with TF-IDF + cosine similarity (scikit-learn)
4. Final score = 60% skill coverage + 40% text similarity

## Tech stack
Python, Streamlit, scikit-learn, pypdf

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Live demo
(add your Streamlit link here)

## Future improvements
- Use NLP embeddings for smarter matching
- Suggest resume improvements
- Support DOCX resumes
