import streamlit as st
import pdfplumber
import spacy
import re
import joblib
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")


class ResumeParser:
    def __init__(self):
        # Initialize with default skills and degrees (can be loaded from model)
        self.SKILLS_DB = [
            "Python", "SQL", "Java", "JavaScript", "R", "C++", "C#", "PHP", "Swift", "Go",
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Keras",
            "Data Analysis", "Data Visualization", "Data Engineering", "Big Data", "Hadoop",
            "Spark", "Pandas", "NumPy", "SciPy", "Scikit-learn", "NLTK", "OpenCV",
            "Power BI", "Tableau", "Looker", "Qlik", "Excel", "Google Data Studio",
            "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQL Server", "Redis", "Cassandra",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD",
            "Git", "GitHub", "GitLab", "JIRA", "Linux", "Bash", "Airflow", "Jenkins"
        ]

        self.DEGREES = [
            "B.Tech", "B.E", "Bachelor", "BS", "BSc", "BA", "B.Com", "BBA", "BCA",
            "M.Tech", "M.E", "Master", "MS", "MSc", "MA", "MBA", "MCA", "PGDM",
            "PhD", "Doctorate", "Postdoc", "Diploma", "Associate Degree"
        ]

        # Try to load trained model if exists
        self.load_model()

    def load_model(self):
        """Load trained model with skills and degrees"""
        try:
            model = joblib.load('resume_model.pkl')
            self.SKILLS_DB = model['SKILLS_DB']
            self.DEGREES = model['DEGREES']
            st.success("AI model loaded successfully!")
        except Exception as e:
            st.warning(f"Using default skills/degrees: {str(e)}")
            self.save_model()  # Save default model

    def save_model(self):
        """Save current skills and degrees as model"""
        model = {
            'SKILLS_DB': self.SKILLS_DB,
            'DEGREES': self.DEGREES
        }
        joblib.dump(model, 'resume_model.pkl')

    def extract_text(self, file):
        """Extract text from PDF resume"""
        try:
            with pdfplumber.open(file) as pdf:
                return "\n".join(page.extract_text() or '' for page in pdf.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None

    def extract_name(self, text):
        """Extract candidate name using NLP"""
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return "Not Found"

    def extract_skills(self, text):
        """Identify skills from text"""
        if not text:
            return []
        text_lower = text.lower()
        return list(OrderedDict.fromkeys(
            skill for skill in self.SKILLS_DB if skill.lower() in text_lower
        ))

    def calculate_experience(self, text):
        """Calculate total work experience in years"""
        if not text:
            return 0.0
        text = text.lower()
        total_months = 0
        total_months += sum(int(y) * 12 for y in re.findall(r'(\d+)\s*(?:years?|yrs?)', text))
        total_months += sum(int(m) for m in re.findall(r'(\d+)\s*(?:months?|mos?)', text))
        return round(total_months / 12, 2)

    def extract_education(self, text):
        """Extract education degrees"""
        if not text:
            return []
        return sorted(list(set(
            degree for degree in self.DEGREES
            if re.search(r'\b' + re.escape(degree) + r'\b', text, re.IGNORECASE)
        )))

    def extract_projects(self, text):
        """Extract project mentions"""
        if not text:
            return ["No projects mentioned"]
        lines = text.split("\n")
        projects = [
            line.strip() for line in lines
            if any(word in line.lower() for word in ["project", "developed", "implemented"])
        ]
        return projects[:3] if projects else ["No projects mentioned"]

    def suggest_role(self, skills):
        """Predict suitable job role based on skills"""
        role_requirements = {
            "Data Analyst": ["SQL", "Excel", "Power BI", "Tableau"],
            "Data Scientist": ["Python", "Machine Learning", "TensorFlow"],
            "Software Engineer": ["Python", "Java", "C++", "JavaScript"],
            "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "CI/CD"],
            "Product Manager": ["Agile", "JIRA", "Product Roadmap"]
        }
        scores = {role: 0 for role in role_requirements}
        for role, req_skills in role_requirements.items():
            for skill in skills:
                if skill in req_skills:
                    scores[role] += 1
        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else "General Professional"

    def calculate_match(self, resume_text, job_desc):
        """Calculate match score between resume and job description"""
        if not resume_text or not job_desc:
            return 0
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_text, job_desc])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return round(similarity * 100, 2)
        except Exception as e:
            st.error(f"Match calculation error: {e}")
            return 0.0

    def parse_resume(self, file, job_description=""):
        """Main function to parse resume and return structured data"""
        text = self.extract_text(file)
        if not text:
            return {"error": "Failed to extract text from PDF"}

        result = {
            "Name": self.extract_name(text),
            "Skills": ", ".join(self.extract_skills(text)),
            "Experience (years)": self.calculate_experience(text),
            "Education": ", ".join(self.extract_education(text)),
            "Projects": " | ".join(self.extract_projects(text)),
            "Suggested Role": self.suggest_role(self.extract_skills(text))
        }

        if job_description:
            result["Match Score (%)"] = self.calculate_match(text, job_description)

        return result


# Streamlit UI
def main():
    st.title("AI Resume Classifier & Skill Extractor")
    st.markdown("Upload a resume PDF and enter a job description to analyze compatibility")

    # File upload
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

    # Job description input
    job_description = st.text_area("Job Description", height=150)

    # Predict button
    if st.button("Analyze Resume"):
        if uploaded_file is not None:
            parser = ResumeParser()
            result = parser.parse_resume(uploaded_file, job_description)

            if "error" in result:
                st.error(result["error"])
            else:
                # Display results in table format
                df = pd.DataFrame.from_dict(result, orient='index', columns=['Value'])
                st.dataframe(df, use_container_width=True)

                # Optional: Show raw JSON
                with st.expander("View Raw JSON Output"):
                    st.json(result)
        else:
            st.warning("Please upload a resume PDF first")


if __name__ == "__main__":
    main()