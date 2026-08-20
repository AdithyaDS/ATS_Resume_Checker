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

DOMAINS = [
    "Computer Science / Software",
    "Data Science / ML / AI",
    "Marketing",
    "Finance",
    "Design",
    "Other",
]


# ---------------- Model loading ----------------

@st.cache_resource
def load_models():
    """Load all heavy models once and cache them across reruns/users."""
    nltk.download("stopwords")
    embed_model = SentenceTransformer("shawhin/distilroberta-ai-job-embeddings")
    nlp = spacy.load("en_core_web_md")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return embed_model, skill_extractor


# ---------------- Text extraction ----------------

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


def extract_resume_text(resume_file) -> str:
    if resume_file.name.endswith(".pdf"):
        return extract_text_from_pdf(resume_file)
    return extract_text_from_docx(resume_file)


# ---------------- Skill extraction & scoring ----------------

def clean_annotations(annotations, min_score: float = 0.65) -> set:
    """Filter skillNer output to real, actionable technical skills.

    Uses three filters: (1) stopwords, (2) confidence score for fuzzy matches,
    (3) skill_type == 'Hard Skill' from skillNer's own database, since soft-skill
    phrases like 'collaboratively' or 'professionally' aren't things you can
    meaningfully add as a resume keyword.
    """
    english_stopwords = set(stopwords.words("english"))
    clean_skills = set()

    def is_hard_skill(skill_id) -> bool:
        entry = SKILL_DB.get(skill_id, {})
        return entry.get("skill_type", "").lower() == "hard skill"

    for match in annotations["results"]["full_matches"]:
        skill = match["doc_node_value"].lower().strip()
        if len(skill) > 2 and skill not in english_stopwords and is_hard_skill(match.get("skill_id")):
            clean_skills.add(skill)

    for match in annotations["results"]["ngram_scored"]:
        skill = match["doc_node_value"].lower().strip()
        score = match.get("score", 0)
        if (
            len(skill) > 2
            and skill not in english_stopwords
            and score >= min_score
            and is_hard_skill(match.get("skill_id"))
        ):
            clean_skills.add(skill)

    # Drop any skill that's just a shorter substring of another matched skill
    # (e.g. "professional" swallowed by "professional development") to reduce
    # near-duplicate variants in the final list.
    deduped = {
        s for s in clean_skills
        if not any(s != other and s in other for other in clean_skills)
    }

    return deduped


def compute_scores(embed_model, skill_extractor, resume_text: str, jd_text: str, weight_keyword: float = 0.6) -> dict:
    """Compute semantic + keyword scores for one resume/JD pair. Reused by both
    the single-JD tab and the multi-job ranking tab."""
    resume_emb = embed_model.encode(resume_text)
    jd_emb = embed_model.encode(jd_text)
    similarity_score = float(
        cosine_similarity(resume_emb.reshape(1, -1), jd_emb.reshape(1, -1))[0][0]
    )

    resume_annotations = skill_extractor.annotate(resume_text)
    jd_annotations = skill_extractor.annotate(jd_text)
    resume_skills = clean_annotations(resume_annotations)
    jd_skills = clean_annotations(jd_annotations)
    matched_skills = jd_skills.intersection(resume_skills)
    missing_skills = jd_skills - resume_skills
    keyword_match_score = len(matched_skills) / len(jd_skills) if jd_skills else 0

    final_score = (keyword_match_score * weight_keyword) + (similarity_score * (1 - weight_keyword))

    return {
        "similarity": similarity_score,
        "keyword": keyword_match_score,
        "final": final_score,
        "matched": matched_skills,
        "missing": missing_skills,
    }


def rating_for(score: float) -> str:
    if score >= 0.75:
        return "Excellent Match"
    elif score >= 0.5:
        return "Good Match — Some Improvements Needed"
    elif score >= 0.3:
        return "Fair Match — Significant Gaps"
    return "Poor Match — Major Revisions Needed"


# ---------------- LLM: full resume analysis ----------------

def get_full_analysis(client, resume_text, jd_text, keyword_score, semantic_score, matched, missing) -> dict:
    """Single LLM call that returns suggestions, parsed resume sections
    (skills + projects), and a hands-on-experience check for JD skills."""
    prompt = f"""You are an expert resume reviewer and ATS analyst. Analyze this resume against the job description and respond ONLY with valid JSON, no markdown, no extra text.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

ANALYSIS RESULTS:
- Keyword Match Score: {keyword_score:.2%}
- Semantic Similarity Score: {semantic_score:.2%}
- Matched Skills: {', '.join(matched) if matched else 'none'}
- Missing Skills: {', '.join(missing) if missing else 'none'}

Respond with this exact JSON structure:
{{
  "overall_feedback": "1-2 sentence summary of fit",
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "missing_skill_fixes": [{{"skill": "skill name", "fix": "how to address it"}}],
  "bullet_rewrite": {{"original": "original bullet text", "improved": "rewritten bullet text"}},
  "skills_section": ["skill1 as written in resume's skills list", "skill2", "..."],
  "projects": [
    {{"title": "project title", "tech_stack": ["tech1", "tech2"], "summary": "brief 2-sentence summary of what it does"}}
  ],
  "hands_on_check": [
    {{"skill": "skill name from JD", "has_hands_on_experience": true, "note": "why - e.g. 'used in Project X' or, if false, a suggestion for how to gain real experience with it"}}
  ]
}}

For "hands_on_check": evaluate each JD-required skill. A skill only listed in a skills list (with no project/experience showing it used) should be marked false, with a practical suggestion (a project idea, or how to reframe existing work). A skill that appears in a project or work experience description should be marked true.
For "projects": extract up to 5 real projects mentioned in the resume, however brief.
Only include information that is actually present or reasonably inferable from the resume - do not invent project names or skills that are not there."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    result_text = response.choices[0].message.content.strip()
    result_text = result_text.replace("```json", "").replace("```", "").strip()
    return json.loads(result_text)


# ---------------- Multi-JD parsing ----------------

def parse_multi_jd(raw_text: str) -> list:
    """Split a pasted block of multiple job descriptions into (title, text) pairs.
    Expected format: blocks separated by a line of '---', each optionally starting
    with 'Title: ...' on its own line."""
    blocks = [b.strip() for b in raw_text.split("---") if b.strip()]
    jobs = []
    for i, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if lines and lines[0].lower().startswith("title:"):
            title = lines[0].split(":", 1)[1].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            title = f"Job {i}"
            body = block
        if body:
            jobs.append((title, body))
    return jobs


# ---------------- UI ----------------

st.title("📄 ATS Resume Matcher")
st.caption(
    "Upload your resume to get an ATS-style match score, AI-powered improvement "
    "suggestions, resume breakdown, and ranking against multiple jobs."
)

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(
        "Groq API Key", type="password", help="Get a free key at console.groq.com"
    )
    domain = st.selectbox("Target Domain", DOMAINS)
    st.caption("Used to label results — matching is still based on your actual resume/JD text.")
    st.markdown("---")
    st.markdown("**Scoring weights**")
    weight_keyword = st.slider("Keyword match weight", 0.0, 1.0, 0.6)
    weight_semantic = 1.0 - weight_keyword
    st.caption(f"Semantic weight: {weight_semantic:.1f}")

st.markdown("### 1. Upload your resume")
resume_file = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])

if resume_file:
    if "resume_name" not in st.session_state or st.session_state.resume_name != resume_file.name:
        with st.spinner("Reading resume..."):
            st.session_state.resume_text = extract_resume_text(resume_file)
            st.session_state.resume_name = resume_file.name

tab1, tab2 = st.tabs(["📋 Match a Single Job Description", "🔍 Rank Against Multiple Jobs"])

# ---------- TAB 1: single JD, full analysis ----------
with tab1:
    jd_text = st.text_area("Paste Job Description", height=220, key="single_jd")

    if st.button("Analyze", type="primary"):
        if not resume_file or not jd_text.strip():
            st.error("Please upload a resume and paste a job description.")
        elif not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            resume_text = st.session_state.resume_text

            with st.spinner("Loading models (first run takes longer)..."):
                embed_model, skill_extractor = load_models()

            with st.spinner("Scoring resume against job description..."):
                scores = compute_scores(embed_model, skill_extractor, resume_text, jd_text, weight_keyword)

            analysis = None
            with st.spinner("Generating AI analysis (suggestions, sections, hands-on check)..."):
                try:
                    client = Groq(api_key=groq_api_key)
                    analysis = get_full_analysis(
                        client, resume_text, jd_text,
                        scores["keyword"], scores["similarity"],
                        scores["matched"], scores["missing"],
                    )
                except Exception as e:
                    st.error(f"Could not generate AI analysis: {e}")

            # Store everything needed to render results (and the projected-score
            # simulator below) so it survives future reruns triggered by other
            # widgets, not just this button click.
            st.session_state.tab1_result = {
                "jd_text": jd_text,
                "scores": scores,
                "analysis": analysis,
            }

    if "tab1_result" in st.session_state:
        result = st.session_state.tab1_result
        scores = result["scores"]
        analysis = result["analysis"]
        jd_text_used = result["jd_text"]
        rating = rating_for(scores["final"])

        st.markdown("## Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Final ATS Score", f"{scores['final']:.0%}")
        m2.metric("Keyword Match", f"{scores['keyword']:.0%}")
        m3.metric("Semantic Similarity", f"{scores['similarity']:.0%}")
        st.markdown(f"**Rating:** {rating}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(sorted(scores["matched"])) if scores["matched"] else "None found")
        with c2:
            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(sorted(scores["missing"])) if scores["missing"] else "None — great match!")

        if analysis:
            st.markdown("## 💡 AI-Powered Suggestions")
            st.info(analysis["overall_feedback"])

            st.markdown("**Actionable Suggestions:**")
            for s in analysis["suggestions"]:
                st.markdown(f"- {s}")

            st.markdown("**How to Address Missing Skills:**")
            for fix in analysis["missing_skill_fixes"]:
                st.markdown(f"- **{fix['skill']}**: {fix['fix']}")

            st.markdown("**Bullet Point Rewrite Example:**")
            st.markdown(f"*Original:* {analysis['bullet_rewrite']['original']}")
            st.markdown(f"*Improved:* {analysis['bullet_rewrite']['improved']}")

            st.markdown("## 🧩 Resume Breakdown")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("### Skills Section")
                skills_list = analysis.get("skills_section", [])
                if skills_list:
                    st.write(", ".join(skills_list))
                else:
                    st.write("No dedicated skills section detected.")

            with b2:
                st.markdown("### Hands-On Experience Check")
                checks = analysis.get("hands_on_check", [])
                if checks:
                    for item in checks:
                        icon = "✅" if item.get("has_hands_on_experience") else "⚠️"
                        st.markdown(f"{icon} **{item['skill']}** — {item['note']}")
                else:
                    st.write("No skills to check.")

            st.markdown("### Projects")
            projects = analysis.get("projects", [])
            if projects:
                for proj in projects:
                    with st.container(border=True):
                        st.markdown(f"**{proj['title']}**")
                        st.caption(", ".join(proj.get("tech_stack", [])))
                        st.write(proj.get("summary", ""))
            else:
                st.write("No distinct projects detected in the resume.")

        # ---- Projected score: "what if I added these skills?" ----
        st.markdown("## 📈 Projected Score")
        missing_list = sorted(scores["missing"])
        if not missing_list:
            st.write("No missing skills — you're already matching everything found in the JD!")
        else:
            st.caption(
                "Pick which missing skills you'd realistically add to your resume, "
                "and see how much your score would improve."
            )
            selected_skills = st.multiselect(
                "Missing skills to simulate adding",
                missing_list,
                default=missing_list,
                key="projected_skills",
            )
            if st.button("Calculate Projected Score", key="calc_projected"):
                if not selected_skills:
                    st.warning("Select at least one skill to simulate.")
                else:
                    with st.spinner("Recalculating with simulated skills..."):
                        embed_model, skill_extractor = load_models()
                        augmented_resume = (
                            st.session_state.resume_text
                            + "\n\nAdditional Skills: "
                            + ", ".join(selected_skills)
                        )
                        projected_scores = compute_scores(
                            embed_model, skill_extractor, augmented_resume, jd_text_used, weight_keyword
                        )

                    p1, p2, p3 = st.columns(3)
                    p1.metric(
                        "Projected Final Score",
                        f"{projected_scores['final']:.0%}",
                        delta=f"{(projected_scores['final'] - scores['final']) * 100:+.1f} pts",
                    )
                    p2.metric(
                        "Projected Keyword Match",
                        f"{projected_scores['keyword']:.0%}",
                        delta=f"{(projected_scores['keyword'] - scores['keyword']) * 100:+.1f} pts",
                    )
                    p3.metric(
                        "Projected Semantic Similarity",
                        f"{projected_scores['similarity']:.0%}",
                        delta=f"{(projected_scores['similarity'] - scores['similarity']) * 100:+.1f} pts",
                    )
                    st.caption(
                        "This simulates adding these skills as plain text (e.g. to a skills "
                        "section) — it does not guarantee a real ATS would score it the same way, "
                        "but it shows the realistic ceiling for how much closing these gaps could help."
                    )

# ---------- TAB 2: multiple JDs, ranked ----------
with tab2:
    st.markdown(
        "Paste multiple job descriptions below, separated by a line containing only `---`. "
        "Optionally start each one with `Title: ...` to label it."
    )
    st.code(
        "Title: Data Scientist at Acme\n<job description text>\n---\nTitle: ML Engineer at Beta\n<job description text>",
        language="text",
    )
    multi_jd_text = st.text_area("Paste multiple Job Descriptions", height=250, key="multi_jd")

    if st.button("Rank Jobs", type="primary"):
        if not resume_file:
            st.error("Please upload a resume above first.")
        elif not multi_jd_text.strip():
            st.error("Please paste at least one job description.")
        else:
            jobs = parse_multi_jd(multi_jd_text)
            if not jobs:
                st.error("Could not find any job descriptions — check the `---` separators.")
            else:
                resume_text = st.session_state.resume_text
                with st.spinner("Loading models (first run takes longer)..."):
                    embed_model, skill_extractor = load_models()

                results = []
                progress = st.progress(0, text="Scoring jobs...")
                for i, (title, jd_body) in enumerate(jobs):
                    scores = compute_scores(embed_model, skill_extractor, resume_text, jd_body, weight_keyword)
                    results.append({
                        "Job": title,
                        "Match %": round(scores["final"] * 100, 1),
                        "Keyword %": round(scores["keyword"] * 100, 1),
                        "Semantic %": round(scores["similarity"] * 100, 1),
                        "Missing Skills": ", ".join(sorted(scores["missing"])) or "None",
                    })
                    progress.progress((i + 1) / len(jobs), text=f"Scored {title}")
                progress.empty()

                results.sort(key=lambda r: r["Match %"], reverse=True)

                st.markdown(f"## Results for **{domain}**")
                st.dataframe(results, use_container_width=True, hide_index=True)