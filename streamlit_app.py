"""
Streamlit Dashboard for Multilingual Content Moderator.
Run with: streamlit run streamlit_app.py
"""

import os
import time
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# Configuration
# ============================================
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_BASE = f"{API_URL}/api/v1"

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Content Moderator",
    page_icon="🛡️",
    layout="wide",
)

# ============================================
# Helper Functions
# ============================================
def check_api_health():
    """Check if the API is running and return health info."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.ConnectionError:
        return None
    return None


def moderate_text(text, threshold=None):
    """Send text to the moderation API and return results."""
    payload = {"text": text}
    if threshold is not None:
        payload["threshold"] = threshold
    try:
        response = requests.post(f"{API_BASE}/moderate", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} — {response.text}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure the server is running.")
        return None


def moderate_batch(texts, threshold=None):
    """Send multiple texts to the batch moderation API."""
    payload = {"texts": texts}
    if threshold is not None:
        payload["threshold"] = threshold
    try:
        response = requests.post(f"{API_BASE}/moderate/batch", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to API.")
        return None


def get_verdict_color(verdict):
    """Return color based on verdict."""
    if verdict == "toxic":
        return "#FF4B4B"  # Red
    elif verdict == "clean":
        return "#00CC66"  # Green
    else:
        return "#FFA500"  # Orange for needs_review


# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.title("🛡️ Settings")

    # API Health
    health = check_api_health()
    if health:
        st.success(f"API: {health['status']}")
        st.caption(f"Model: {health['model_name'].split('/')[-1]}")
        st.caption(f"Device: {health['device']}")
    else:
        st.error("API is offline")
        st.caption("Start the API with:")
        st.code("uvicorn app.main:app --port 8000")

    st.divider()

    # Threshold control
    use_custom_threshold = st.checkbox("Override threshold", value=False)
    if use_custom_threshold:
        custom_threshold = st.slider(
            "Custom threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Override the language-aware default threshold",
        )
    else:
        custom_threshold = None
        st.caption("Using language-aware defaults:")
        st.caption("EN: 0.50 | HI: 0.15 | AR: 0.20")

    st.divider()

    # Supported Languages
    st.subheader("Supported Languages")
    st.caption("🇬🇧 English  •  🇮🇳 Hindi  •  🇸🇦 Arabic")

    st.divider()

    # History control
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ============================================
# Initialize Session State
# ============================================
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================
# Main Content
# ============================================
st.title("🛡️ Multilingual Content Moderator")
st.markdown("Detect toxicity across **English**, **Hindi**, and **Arabic** using AI")

# ============================================
# Tab Layout
# ============================================
tab1, tab2 = st.tabs(["✍️ Single Text", "📁 Batch Upload"])

# ============================================
# Tab 1: Single Text Moderation
# ============================================
with tab1:
    # Example text buttons
    st.markdown("**Try an example:**")
    example_cols = st.columns(6)
    examples = [
        ("🇬🇧 Toxic", "You are a disgusting and terrible person"),
        ("🇬🇧 Clean", "Thank you for your wonderful help today!"),
        ("🇮🇳 Toxic", "तुम एक बेवकूफ और बेकार इंसान हो"),
        ("🇮🇳 Clean", "आपका काम बहुत अच्छा है"),
        ("🇸🇦 Toxic", "أنت غبي ولا تستحق الاحترام"),
        ("🇸🇦 Clean", "شكراً لك على مساعدتك"),
    ]

    # Track which example was clicked
    selected_example = None
    for i, (label, text) in enumerate(examples):
        with example_cols[i]:
            if st.button(label, use_container_width=True, key=f"example_{i}"):
                selected_example = text

    # Text input
    default_text = selected_example if selected_example else ""
    if selected_example:
        st.session_state["text_input"] = selected_example

    text_input = st.text_area(
        "Enter text to moderate:",
        value=st.session_state.get("text_input", ""),
        height=100,
        placeholder="Type or paste text here...",
        key="text_input",
    )

    # Analyse button
    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not text_input or not text_input.strip():
            st.warning("Please enter some text to analyse.")
        else:
            with st.spinner("Analysing text..."):
                result = moderate_text(text_input, threshold=custom_threshold)

            if result:
                # Store in history
                st.session_state.history.insert(0, result)

                # Display result
                st.divider()

                # Verdict banner
                verdict = result["verdict"]
                color = get_verdict_color(verdict)
                confidence = result.get("confidence", 0)
                toxic_score = result.get("categories", {}).get("toxic", {}).get("score", 0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f"<div style='background-color:{color}; padding:20px; "
                        f"border-radius:10px; text-align:center;'>"
                        f"<h2 style='color:white; margin:0;'>{verdict.upper()}</h2></div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.metric("Toxic Score", f"{toxic_score:.1%}")
                with col3:
                    st.metric("Language", result["language"].upper())
                with col4:
                    threshold_used = result.get("threshold_used", "N/A")
                    if isinstance(threshold_used, (int, float)):
                        st.metric("Threshold", f"{threshold_used:.2f}")
                    else:
                        st.metric("Threshold", threshold_used)

                # Warnings
                warnings = result.get("warnings", [])
                if warnings:
                    for w in warnings:
                        st.warning(w)

                # Category scores chart
                categories = result.get("categories", {})
                if categories:
                    st.subheader("Category Scores")
                    cat_names = list(categories.keys())
                    cat_scores = [categories[c]["score"] for c in cat_names]
                    cat_flagged = [categories[c]["flagged"] for c in cat_names]
                    cat_colors = ["#FF4B4B" if f else "#00CC66" for f in cat_flagged]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=cat_names,
                            y=cat_scores,
                            marker_color=cat_colors,
                            text=[f"{s:.1%}" for s in cat_scores],
                            textposition="auto",
                        )
                    ])
                    fig.update_layout(
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=300,
                        margin=dict(t=10, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Processing time
                proc_time = result.get("processing_time_ms", 0)
                st.caption(f"Processed in {proc_time:.0f}ms")


# ============================================
# Tab 2: Batch Upload
# ============================================
with tab2:
    st.markdown("Upload a CSV file with a **text** column to moderate multiple texts at once.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            # Find the text column
            text_col = None
            for col in df.columns:
                if col.lower() in ["text", "content", "message", "comment"]:
                    text_col = col
                    break

            if text_col is None:
                st.error(
                    f"No text column found. Your CSV has columns: {list(df.columns)}. "
                    f"Please rename one to 'text'."
                )
            else:
                st.success(f"Found {len(df)} texts in column '{text_col}'")
                st.dataframe(df[[text_col]].head(5), use_container_width=True)

                if st.button("🚀 Moderate All", type="primary", use_container_width=True):
                    texts = df[text_col].dropna().astype(str).tolist()

                    with st.spinner(f"Moderating {len(texts)} texts..."):
                        batch_result = moderate_batch(texts, threshold=custom_threshold)

                    if batch_result:
                        st.divider()

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Texts", batch_result["total_texts"])
                        with col2:
                            st.metric("Flagged", batch_result["flagged_count"])
                        with col3:
                            st.metric("Clean", batch_result["clean_count"])
                        with col4:
                            total_time = batch_result.get("total_processing_time_ms", 0)
                            st.metric("Total Time", f"{total_time:.0f}ms")

                        # Build results dataframe
                        results_data = []
                        for r in batch_result["results"]:
                            toxic_score = r.get("categories", {}).get("toxic", {}).get("score", 0)
                            results_data.append({
                                "Text": r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"],
                                "Language": r["language"],
                                "Verdict": r["verdict"],
                                "Toxic Score": round(toxic_score, 4),
                                "Threshold": r.get("threshold_used", "N/A"),
                            })
                        results_df = pd.DataFrame(results_data)

                        # Results table
                        st.subheader("Results")
                        st.dataframe(
                            results_df.style.map(
                                lambda v: "color: #FF4B4B" if v == "toxic" else "color: #00CC66",
                                subset=["Verdict"],
                            ),
                            use_container_width=True,
                        )

                        # Charts side by side
                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            # Verdict distribution pie chart
                            st.subheader("Verdict Distribution")
                            verdict_counts = results_df["Verdict"].value_counts()
                            fig_pie = px.pie(
                                values=verdict_counts.values,
                                names=verdict_counts.index,
                                color=verdict_counts.index,
                                color_discrete_map={"toxic": "#FF4B4B", "clean": "#00CC66"},
                            )
                            fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with chart_col2:
                            # Language distribution bar chart
                            st.subheader("Language Distribution")
                            lang_counts = results_df["Language"].value_counts()
                            fig_lang = px.bar(
                                x=lang_counts.index,
                                y=lang_counts.values,
                                labels={"x": "Language", "y": "Count"},
                                color=lang_counts.index,
                            )
                            fig_lang.update_layout(
                                height=300,
                                margin=dict(t=10, b=10),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_lang, use_container_width=True)

                        # Download results
                        csv_download = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_download,
                            file_name="moderation_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )


# ============================================
# History Section
# ============================================
if st.session_state.history:
    st.divider()
    st.subheader("📜 Recent History")

    history_data = []
    for r in st.session_state.history[:20]:  # Show last 20
        toxic_score = r.get("categories", {}).get("toxic", {}).get("score", 0)
        history_data.append({
            "Text": r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"],
            "Language": r["language"],
            "Verdict": r["verdict"],
            "Toxic Score": round(toxic_score, 4),
            "Threshold": r.get("threshold_used", "N/A"),
        })

    history_df = pd.DataFrame(history_data)
    st.dataframe(
        history_df.style.map(
            lambda v: "color: #FF4B4B" if v == "toxic" else "color: #00CC66",
            subset=["Verdict"],
        ),
        use_container_width=True,
    )