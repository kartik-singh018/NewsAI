import streamlit as st
import joblib
import os
import sys
import html


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NewsAI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "news_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

SRC_PATH = os.path.join(
    BASE_DIR,
    "src"
)

sys.path.insert(0, SRC_PATH)

from preprocessing import preprocess_text


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_models()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================
   GLOBAL
========================= */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.16), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.13), transparent 30%),
        #080d18;
    color: #f8fafc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================
   HEADER
========================= */

.hero {
    text-align: center;
    padding: 42px 25px 35px 25px;
    margin-bottom: 28px;
}

.hero-icon {
    font-size: 55px;
    margin-bottom: 5px;
}

.hero-title {
    font-size: 54px;
    font-weight: 800;
    letter-spacing: -2px;
    color: #ffffff;
    margin: 0;
}

.hero-subtitle {
    margin-top: 10px;
    color: #94a3b8;
    font-size: 18px;
}

.hero-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(96,165,250,0.25);
    color: #93c5fd;
    font-size: 14px;
}


/* =========================
   STAT CARDS
========================= */

.stat-card {
    background: linear-gradient(
        145deg,
        rgba(30,41,59,0.9),
        rgba(15,23,42,0.9)
    );

    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 18px;
    padding: 24px 18px;
    text-align: center;
    min-height: 125px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.18);
}

.stat-icon {
    font-size: 25px;
}

.stat-number {
    font-size: 27px;
    font-weight: 800;
    color: #60a5fa;
    margin-top: 7px;
}

.stat-label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 3px;
}


/* =========================
   SECTION HEADINGS
========================= */

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 35px;
    margin-bottom: 14px;
    color: #f8fafc;
}


/* =========================
   TEXT AREA
========================= */

.stTextArea textarea {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    padding: 18px !important;
}

.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}


/* =========================
   BUTTONS
========================= */

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: #111827 !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    transition: 0.2s ease !important;
}

.stButton > button:hover {
    border-color: #60a5fa !important;
    color: #ffffff !important;
    background: #172554 !important;
    transform: translateY(-1px);
}

div[data-testid="stFormSubmitButton"] button,
button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    border: none !important;
    color: white !important;
    font-size: 17px !important;
    font-weight: 750 !important;
    padding: 12px 25px !important;
}


/* =========================
   RESULT
========================= */

.result-card {
    margin-top: 30px;
    padding: 35px;
    border-radius: 22px;

    background:
        radial-gradient(
            circle at top right,
            rgba(99,102,241,0.25),
            transparent 45%
        ),
        linear-gradient(
            145deg,
            #172554,
            #111827
        );

    border: 1px solid rgba(96,165,250,0.25);
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

.result-label {
    color: #93c5fd;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-category {
    font-size: 40px;
    font-weight: 850;
    margin: 12px 0;
    color: white;
}

.result-confidence {
    color: #cbd5e1;
    font-size: 17px;
}

.confidence-number {
    color: #60a5fa;
    font-weight: 800;
}


/* =========================
   PROBABILITY
========================= */

.probability-card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
}

.probability-header {
    display: flex;
    justify-content: space-between;
    font-weight: 650;
    color: #e2e8f0;
    margin-bottom: 9px;
}


/* =========================
   PIPELINE
========================= */

.pipeline-card {
    background: #111827;
    border: 1px solid #263449;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    min-height: 115px;
}

.pipeline-number {
    font-size: 13px;
    color: #60a5fa;
    font-weight: 800;
}

.pipeline-name {
    margin-top: 8px;
    color: #e2e8f0;
    font-weight: 650;
}


/* =========================
   FOOTER
========================= */

.footer {
    text-align: center;
    margin-top: 55px;
    padding-top: 25px;
    border-top: 1px solid #1e293b;
    color: #64748b;
    font-size: 14px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-icon">📰</div>
    <div class="hero-title">NewsAI</div>
    <div class="hero-subtitle">
        Intelligent News Categorization using Natural Language Processing
    </div>
    <div class="hero-badge">
        ✨ TF-IDF + Multinomial Naive Bayes
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# STATISTICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
<div class="stat-card">
    <div class="stat-icon">🗞️</div>
    <div class="stat-number">127.6K</div>
    <div class="stat-label">Training Articles</div>
</div>
""",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
<div class="stat-card">
    <div class="stat-icon">🗂️</div>
    <div class="stat-number">4</div>
    <div class="stat-label">News Categories</div>
</div>
""",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
<div class="stat-card">
    <div class="stat-icon">🎯</div>
    <div class="stat-number">89.95%</div>
    <div class="stat-label">Model Accuracy</div>
</div>
""",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        """
<div class="stat-card">
    <div class="stat-icon">🧠</div>
    <div class="stat-number">NLP</div>
    <div class="stat-label">Technology</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# ARTICLE INPUT
# ============================================================

st.markdown(
    '<div class="section-title">✍️ Analyze a News Article</div>',
    unsafe_allow_html=True
)

article = st.text_area(
    "News article",
    height=220,
    placeholder=(
        "Paste a news article here...\n\n"
        "For example: The football team won the championship "
        "after an exciting final match..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# EXAMPLE BUTTONS
# ============================================================

st.markdown(
    '<div style="color:#94a3b8;margin:8px 0 12px 0;">⚡ Try an example</div>',
    unsafe_allow_html=True
)

e1, e2, e3, e4 = st.columns(4)

if "example_text" not in st.session_state:
    st.session_state.example_text = ""


with e1:
    if st.button("💻 Technology", use_container_width=True):
        st.session_state.example_text = (
            "The company announced a new artificial intelligence "
            "system that can process large amounts of data."
        )
        st.rerun()

with e2:
    if st.button("⚽ Sports", use_container_width=True):
        st.session_state.example_text = (
            "The football team defeated its opponent in the final "
            "match and won the championship."
        )
        st.rerun()

with e3:
    if st.button("💰 Business", use_container_width=True):
        st.session_state.example_text = (
            "The stock market rose today as investors reacted to "
            "strong company earnings and economic growth."
        )
        st.rerun()

with e4:
    if st.button("🌍 World", use_container_width=True):
        st.session_state.example_text = (
            "World leaders met today to discuss international "
            "relations and a new agreement between several countries."
        )
        st.rerun()


# Use example if selected
if st.session_state.example_text:
    article = st.session_state.example_text

    st.info(
        "Example article loaded. Click **CLASSIFY ARTICLE** to analyze it."
    )


# ============================================================
# CLASSIFY
# ============================================================

st.write("")

classify = st.button(
    "🚀  CLASSIFY ARTICLE",
    type="primary",
    use_container_width=True
)


if classify:

    if not article.strip():

        st.warning("⚠️ Please enter or select a news article first.")

    else:

        with st.spinner("🧠 Processing article with NLP..."):

            clean_text = preprocess_text(article)

            text_vector = vectorizer.transform(
                [clean_text]
            )

            prediction = model.predict(
                text_vector
            )[0]

            probabilities = model.predict_proba(
                text_vector
            )[0]

            classes = model.classes_

            probability_dict = dict(
                zip(classes, probabilities)
            )

            confidence = probability_dict[prediction] * 100


        icons = {
            "Business": "💰",
            "Sci/Tech": "💻",
            "Sports": "⚽",
            "World": "🌍"
        }

        icon = icons.get(
            prediction,
            "📰"
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            f"""
<div class="result-card">
    <div class="result-label">PREDICTED CATEGORY</div>
    <div class="result-category">
        {icon} {html.escape(str(prediction))}
    </div>
    <div class="result-confidence">
        Confidence:
        <span class="confidence-number">
            {confidence:.2f}%
        </span>
    </div>
</div>
""",
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Breakdown</div>',
            unsafe_allow_html=True
        )

        for category in classes:

            probability = probability_dict[category] * 100

            category_icon = icons.get(
                category,
                "📰"
            )

            st.markdown(
                f"""
<div class="probability-card">
    <div class="probability-header">
        <span>{category_icon} {html.escape(str(category))}</span>
        <span>{probability:.2f}%</span>
    </div>
</div>
""",
                unsafe_allow_html=True
            )

            st.progress(
                float(probability) /100
            )


        # ====================================================
        # PIPELINE
        # ====================================================

        st.markdown(
            '<div class="section-title">🔍 NLP Processing Pipeline</div>',
            unsafe_allow_html=True
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.markdown(
                """
<div class="pipeline-card">
    <div class="pipeline-number">STEP 01</div>
    <div class="pipeline-name">🧹 Text Cleaning</div>
</div>
""",
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                """
<div class="pipeline-card">
    <div class="pipeline-number">STEP 02</div>
    <div class="pipeline-name">🔤 Tokenization</div>
</div>
""",
                unsafe_allow_html=True
            )

        with p3:
            st.markdown(
                """
<div class="pipeline-card">
    <div class="pipeline-number">STEP 03</div>
    <div class="pipeline-name">📊 TF-IDF</div>
</div>
""",
                unsafe_allow_html=True
            )

        with p4:
            st.markdown(
                """
<div class="pipeline-card">
    <div class="pipeline-number">STEP 04</div>
    <div class="pipeline-name">🤖 Naive Bayes</div>
</div>
""",
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    🧠 NewsAI &nbsp; • &nbsp;
    TF-IDF + Naive Bayes &nbsp; • &nbsp;
    Built with Python & Streamlit
</div>
""",
    unsafe_allow_html=True
)