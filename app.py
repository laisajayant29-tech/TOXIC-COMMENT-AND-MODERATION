from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent
MODEL_PATH = "toxic_comment_model.joblib"
METRICS_PATH = "metrics.json"

st.set_page_config(page_title="CommentCare | NLP Moderation", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main {background: #f7f8fc;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#172554,#334155); color: white;}
.hero h1 {color:white; margin-bottom:.25rem;}
.small-muted {color:#64748b;}
.result-safe {padding:1rem; border-radius:12px; background:#dcfce7; color:#14532d;}
.result-review {padding:1rem; border-radius:12px; background:#ffedd5; color:#7c2d12;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.markdown("""
<div class="hero">
  <h1>🛡️ CommentCare</h1>
  <p>AI-assisted toxic comment detection and human review</p>
  <p style="opacity:.85">A TF-IDF + Logistic Regression NLP demonstration</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("The model has not been trained yet. Run `python train_model.py` from the project folder, then restart the app.")
    st.stop()

model = load_model()
classes = list(model.named_steps["classifier"].classes_)
toxic_idx = classes.index("Toxic")

with st.sidebar:
    st.header("Moderation settings")
    threshold = st.slider("Toxic review threshold", 0.10, 0.95, 0.50, 0.05)
    st.caption("Lower thresholds flag more comments for review, which may also increase false positives.")
    st.divider()
    st.markdown("**How it works**")
    st.write("Comment → TF-IDF features → Logistic Regression → Review recommendation")
    st.warning("This tool is a prototype. It does not make final moderation decisions.")

tab1, tab2, tab3 = st.tabs(["🔎 Analyse a comment", "📊 Model evaluation", "ℹ️ About & limitations"])

with tab1:
    st.subheader("Check a comment")
    comment = st.text_area("Enter text to analyse", height=140, placeholder="Write or paste a comment here...")
    if st.button("Analyse comment", type="primary", use_container_width=True):
        if not comment.strip():
            st.warning("Please enter a comment first.")
        else:
            probabilities = model.predict_proba([comment])[0]
            toxic_score = float(probabilities[toxic_idx])
            predicted = "Toxic" if toxic_score >= threshold else "Safe"
            st.subheader("Analysis result")
            c1, c2 = st.columns(2)
            c1.metric("Predicted category", predicted)
            c2.metric("Toxicity score", f"{toxic_score:.1%}")
            if predicted == "Toxic":
                st.markdown('<div class="result-review"><b>⚠️ Needs human review</b><br>This comment crossed the selected review threshold. Do not automatically punish or remove the author.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-safe"><b>✓ Not flagged by this model</b><br>This is not a guarantee that the comment is harmless.</div>', unsafe_allow_html=True)
            st.progress(min(max(toxic_score, 0.0), 1.0), text="Model toxicity score")
            st.caption("The score is a model estimate, not a calibrated probability or a judgement of intent.")

with tab2:
    st.subheader("Held-out evaluation")
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        a,b,c,d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.1%}")
        b.metric("Toxic precision", f"{m['toxic_precision']:.1%}")
        c.metric("Toxic recall", f"{m['toxic_recall']:.1%}")
        d.metric("Toxic F1", f"{m['toxic_f1']:.1%}")
        cm = pd.DataFrame(m["confusion_matrix"], index=["Actual Safe","Actual Toxic"], columns=["Predicted Safe","Predicted Toxic"])
        st.write("Confusion matrix")
        st.dataframe(cm, use_container_width=True)
        st.caption(m["note"])
    else:
        st.info("Run `python train_model.py` to generate evaluation metrics.")

with tab3:
    st.subheader("Project information")
    st.write("""
    **Purpose:** Demonstrate a basic text moderation workflow using supervised machine learning.

    **Features:** Word and character TF-IDF, Logistic Regression, a configurable review threshold,
    held-out evaluation, and a human-review recommendation.
    """)
    st.warning("""
    **Important limitations**
    - The included dataset is synthetic and small. It is suitable for demonstrating the code flow, not real-world moderation.
    - Toxicity is contextual. Sarcasm, reclaimed language, quotations, dialects and targeted abuse can be misclassified.
    - A low or high score is not proof of intent, factuality or harm.
    - Do not use this prototype for automatic bans, disciplinary decisions or high-stakes decisions.
    - Before deployment, use a large, representative, properly licensed and human-annotated dataset; test subgroup error rates and provide an appeal process.
    """)
    st.write("**Human-in-the-loop:** The model only recommends whether a comment should be reviewed. A trained moderator makes the final decision.")
