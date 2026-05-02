import streamlit as st
import pdfplumber
from transformers import BartForConditionalGeneration, BartTokenizer

import streamlit as st
import pdfplumber

st.set_page_config(page_title="Legal Summarizer")

st.title("📄 Legal Document Summarizer")

st.markdown("### Option 1: Upload PDF")
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

st.markdown("---")

st.markdown("### Option 2: Paste text (for mobile users)")
manual_text = st.text_area("Paste your document text here")

text = ""

# 📂 If file uploaded
if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

# ✍️ If text pasted
elif manual_text:
    text = manual_text

# ✅ Show result
if text:
    st.success("Content loaded ✅")
    st.text_area("Output", text, height=300)

# Custom CSS (🔥 modern UI)
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
.upload-box {
    border: 2px dashed #38bdf8;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚖️ AI Legal Document Summarizer")
st.caption("Upload legal PDFs • Get instant summaries • Save hours of reading ⏱️")

# Sidebar (🔥 PRO FEATURE)
st.sidebar.title("⚙️ Settings")
summary_length = st.sidebar.slider("Summary Length", 50, 300, 150)
show_text = st.sidebar.checkbox("Show Extracted Text", True)

# Load model
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()

# File upload
st.markdown('<div class="upload-box">📂 Upload your Legal PDF</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"])

# Extract text
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Summarize (🔥 improved)
def summarize_text(text):
    inputs = tokenizer(text[:1024], return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=summary_length,
        min_length=50,
        do_sample=False
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Main UI
if uploaded_file:
    text = extract_text(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        if show_text:
            st.subheader("📜 Extracted Text")
            st.text_area("", text[:2000], height=400)

    with col2:
        st.subheader("🧠 AI Summary")

        if st.button("⚡ Generate Summary"):
            with st.spinner("Analyzing document..."):
                summary = summarize_text(text)

                st.success("Summary Ready ✅")
                st.write(summary)

                # 🔥 Download feature
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name="summary.txt"
                )
