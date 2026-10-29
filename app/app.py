"""Step 8: Streamlit demo. The UI only; all logic is in detector.py.

Run from the repository root:   streamlit run app/app.py
Model choice (see detector.py): KHMER_DETECTOR_MODEL=baseline|xlmr, KHMER_DETECTOR_MODEL_PATH=<file, folder or Hub id>
"""
import streamlit as st

st.set_page_config(page_title="Khmer AI-Text Detector", layout="centered")

try:
    import detector
except ImportError as exc:  # src/clean.py is missing next to the app
    st.error(f"The app cannot start: {exc}")
    st.stop()

# Shown in the "About" notes. Update both when the model is retrained on the full data.
TRAINED_ON_PARTIAL_DATA = True
GENERATORS = ("gemini-3.1-flash-lite", "gemma-4-31b-it", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")
HELD_OUT_GENERATOR = "qwen/qwen3.8-27b"

# Khmer glyphs: the page font has none, so a browser without a Khmer system font would show boxes
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Khmer:wght@400;600&display=swap');
    .stApp, textarea { font-family: "Source Sans", "Source Sans Pro", "Segoe UI", "Noto Sans Khmer", "Khmer UI", "Leelawadee UI", sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model...")
def get_model(kind: str, path: str) -> "detector.LoadResult":
    return detector.try_load(detector.ModelConfig(kind, path))


@st.cache_data
def get_examples() -> dict:
    return detector.load_examples()


def fill_with_example(key: str) -> None:
    st.session_state["text"] = get_examples()[key]["text"]


def show_notices(prepared: "detector.Prepared") -> None:
    for notice in prepared.notices:
        st.warning(notice.message)


def show_prediction(prediction: "detector.Prediction") -> None:
    color = "orange" if prediction.is_ai else "blue"
    st.markdown(f"### :{color}[{prediction.verdict}]")
    st.metric("Probability that the text is AI-generated", f"{prediction.p_ai:.1%}")
    st.progress(prediction.p_ai)
    st.caption(f"The verdict switches at {detector.THRESHOLD:.0%}. {prediction.note}")


def show_explanation(explanation: "detector.Explanation") -> None:
    st.markdown("**Why? The character sequences that pushed the score**")
    st.caption(
        "The baseline reads overlapping sequences of 1 to 4 characters. “Push” is how strongly a sequence "
        "moved the score (TF-IDF value × model weight): positive pushes toward AI, negative toward human. "
        "Single characters and overlapping variants are left out. This shows what the model used, "
        "which can be a spelling or punctuation habit rather than meaning."
    )
    left, right = st.columns(2)
    for column, title, items in (
        (left, "Toward AI", explanation.toward_ai),
        (right, "Toward human", explanation.toward_human),
    ):
        column.markdown(f"**{title}**")
        if items:
            column.dataframe(
                [{"Sequence": c.shown, "Push": round(c.weight, 3), "Times in text": c.count} for c in items],
                hide_index=True,
            )
        else:
            column.caption("None in this text.")


def show_about(backend: "detector.Backend | None") -> None:
    st.divider()
    st.subheader("About this demo")
    data_note = (
        "The model you see here is trained on **partial data** (about 1,400 training texts) and will be retrained "
        "when text generation finishes. "
        if TRAINED_ON_PARTIAL_DATA
        else ""
    )
    st.markdown(
        f"""
- **A research demo, not proof.** The score says how much the text resembles the texts the model learned from. It cannot show who wrote a text.
- **Training data.** Human text: Khmer Wikipedia (a November 2023 dump) and mC4 Khmer web news (crawled 2013 to 2020). AI text: written by four language models, {", ".join(GENERATORS)}; the last one ({HELD_OUT_GENERATOR}) was kept out of training to test unseen models. {data_note}
- **It can be wrong.** It is less reliable on short texts (under 300 characters), on other kinds of writing (chat, social media, poetry, translated or heavily edited text), and on AI models it never saw.
- **Do not use it to accuse anyone** of using AI, whether a student, a writer or a journalist.
"""
    )
    if backend is not None:
        st.caption(f"Model in use: {backend.title}.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Khmer AI-Text Detector")
st.caption("ឧបករណ៍ពិនិត្យអត្ថបទខ្មែរ៖ មនុស្សសរសេរ ឬ AI បង្កើត?")
st.info("Paste Khmer text to see whether it looks human-written or AI-generated. This is a research demo, not proof.")

config = detector.config_from_env()
loaded = get_model(config.kind, config.path)
if loaded.backend is None:
    st.error(loaded.error)
    show_about(None)
    st.stop()
backend = loaded.backend
examples = get_examples()

st.session_state.setdefault("text", "")
st.caption("No text to hand? Fill the box with an example:")
human_col, ai_col = st.columns(2)
human_col.button("Human-written example", on_click=fill_with_example, args=("human",))
ai_col.button("AI-generated example", on_click=fill_with_example, args=("ai",))

with st.form("check"):
    st.text_area(
        "Khmer text",
        key="text",
        height=240,
        max_chars=detector.MAX_INPUT_CHARS,
        placeholder="បិទភ្ជាប់អត្ថបទខ្មែរនៅទីនេះ",
    )
    st.form_submit_button("Check text", type="primary")

text = st.session_state["text"]
for key, example in examples.items():  # attribution travels with the example text
    if text.strip() == example["text"].strip():
        st.caption(example["attribution"])

prepared = detector.prepare(text)
if prepared.empty:
    st.info("Nothing to check yet. Paste some Khmer text above, or try an example, then press “Check text”.")
else:
    show_notices(prepared)
    show_prediction(detector.predict(backend, prepared))
    st.caption(f"Scored text: {prepared.n_chars:,} characters, {prepared.khmer_share:.0%} Khmer.")
    explanation = detector.explain(backend, prepared)
    if explanation is not None:
        with st.expander("Why this score?", expanded=True):
            show_explanation(explanation)
    if prepared.removed_chars:
        with st.expander(f"Text the model scored ({prepared.removed_chars:,} characters were cleaned away)"):
            st.caption(
                "The same cleaning as for the training texts was applied first: headline and menu lines, footers, "
                "list markers and anything after the last sentence end are removed."
            )
            st.text(prepared.text)
show_about(backend)
