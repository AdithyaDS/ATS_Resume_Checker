import streamlit as st
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from spacy.matcher import PhraseMatcher
from nltk.corpus import stopwords
import nltk
import json
from groq import Groq

st.set_page_config(page_title="ATS Resume Matcher", layout="wide")


@st.cache_resource
def load_models():
    """Load all heavy models once and cache them across reruns/users."""
    nltk.download("stopwords")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_md")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return embed_model, skill_extractor


def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def clean_annotations(annotations, min_score: float = 0.65) -> set:
    """Filter skillNer output using stopwords + confidence score instead of a manual blacklist."""
    english_stopwords = set(stopwords.words("english"))
    clean_skills = set()

    for match in annotations["results"]["full_matches"]:
        skill = match["doc_node_value"].lower().strip()
        if len(skill) > 2 and skill not in english_stopwords:
            clean_skills.add(skill)

    for match in annotations["results"]["ngram_scored"]:
        skill = match["doc_node_value"].lower().strip()
        score = match.get("score", 0)
        if len(skill) > 2 and skill not in english_stopwords and score >= min_score:
            clean_skills.add(skill)

    return clean_skills


def get_suggestions(client, resume_text, jd_text, keyword_score, semantic_score, matched, missing) -> dict:
    prompt = f"""You are an expert resume reviewer. Analyze this resume against the job description and respond ONLY with valid JSON, no markdown, no extra text.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

ANALYSIS RESULTS:
- Keyword Match Score: {keyword_score:.2%}
- Semantic Similarity Score: {semantic_score:.2%}
- Matched Skills: {', '.join(matched)}
- Missing Skills: {', '.join(missing)}

Respond with this exact JSON structure:
{{
  "overall_feedback": "1-2 sentence summary of fit",
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "missing_skill_fixes": [{{"skill": "skill name", "fix": "how to address it"}}],
  "bullet_rewrite": {{"original": "original bullet text", "improved": "rewritten bullet text"}}
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    result_text = response.choices[0].message.content.strip()
    result_text = result_text.replace("```json", "").replace("```", "").strip()
    return json.loads(result_text)


# ---------------- UI ----------------

st.title("📄 ATS Resume Matcher")
st.caption(
    "Upload your resume and a job description to get an ATS-style match score "
    "and AI-powered improvement suggestions."
)

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(
        "Groq API Key", type="password", help="Get a free key at console.groq.com"
    )
    st.markdown("---")
    st.markdown("**Scoring weights**")
    weight_keyword = st.slider("Keyword match weight", 0.0, 1.0, 0.6)
    weight_semantic = 1.0 - weight_keyword
    st.caption(f"Semantic weight: {weight_semantic:.1f}")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
with col2:
    jd_text = st.text_area("Paste Job Description", height=250)

if st.button("Analyze", type="primary"):
    if not resume_file or not jd_text.strip():
        st.error("Please upload a resume and paste a job description.")
    elif not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Loading models (first run takes longer)..."):
            embed_model, skill_extractor = load_models()

        if resume_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_docx(resume_file)

        with st.spinner("Computing semantic similarity..."):
            resume_emb = embed_model.encode(resume_text)
            jd_emb = embed_model.encode(jd_text)
            similarity_score = cosine_similarity(
                resume_emb.reshape(1, -1), jd_emb.reshape(1, -1)
            )[0][0]

        with st.spinner("Extracting skills..."):
            resume_annotations = skill_extractor.annotate(resume_text)
            jd_annotations = skill_extractor.annotate(jd_text)
            resume_skills = clean_annotations(resume_annotations)
            jd_skills = clean_annotations(jd_annotations)
            matched_skills = jd_skills.intersection(resume_skills)
            missing_skills = jd_skills - resume_skills
            keyword_match_score = len(matched_skills) / len(jd_skills) if jd_skills else 0

        final_ats_score = (keyword_match_score * weight_keyword) + (similarity_score * weight_semantic)

        if final_ats_score >= 0.75:
            rating = "Excellent Match"
        elif final_ats_score >= 0.5:
            rating = "Good Match — Some Improvements Needed"
        elif final_ats_score >= 0.3:
            rating = "Fair Match — Significant Gaps"
        else:
            rating = "Poor Match — Major Revisions Needed"

        st.markdown("## Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Final ATS Score", f"{final_ats_score:.0%}")
        m2.metric("Keyword Match", f"{keyword_match_score:.0%}")
        m3.metric("Semantic Similarity", f"{similarity_score:.0%}")
        st.markdown(f"**Rating:** {rating}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(sorted(matched_skills)) if matched_skills else "None found")
        with c2:
            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(sorted(missing_skills)) if missing_skills else "None — great match!")

        with st.spinner("Generating AI suggestions..."):
            try:
                client = Groq(api_key=groq_api_key)
                suggestions = get_suggestions(
                    client,
                    resume_text,
                    jd_text,
                    keyword_match_score,
                    similarity_score,
                    matched_skills,
                    missing_skills,
                )

                st.markdown("## 💡 AI-Powered Suggestions")
                st.info(suggestions["overall_feedback"])

                st.markdown("**Actionable Suggestions:**")
                for s in suggestions["suggestions"]:
                    st.markdown(f"- {s}")

                st.markdown("**How to Address Missing Skills:**")
                for fix in suggestions["missing_skill_fixes"]:
                    st.markdown(f"- **{fix['skill']}**: {fix['fix']}")

                st.markdown("**Bullet Point Rewrite Example:**")
                st.markdown(f"*Original:* {suggestions['bullet_rewrite']['original']}")
                st.markdown(f"*Improved:* {suggestions['bullet_rewrite']['improved']}")
            except Exception as e:
                st.error(f"Could not generate AI suggestions: {e}")
