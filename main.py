import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

st.title("AI Resume Ranker (Top 10)")

# 1. Inputs
job_description = st.text_area("Paste the Job Description here")
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes") and job_description and uploaded_files:
    results = []
    
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        
        # 2. Scoring
        corpus = [job_description, resume_text]
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0] # 0.49088650210868884
        results.append({"Name": file.name, "Score": round(score * 100, 2)}) 
    
    # 3. Sorting and Displaying Top 10
    top_10 = sorted(results, key=lambda x: x['Score'], reverse=True)[:10]
    st.table(top_10)