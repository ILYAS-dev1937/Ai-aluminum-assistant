"""app.py — Streamlit chat interface for MAVAL AI Aluminum Assistant."""
import streamlit as st
from retriever import MavalRetriever
from assistant import MavalAssistant

st.set_page_config(
    page_title="MAVAL AI Assistant",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ MAVAL AI Aluminum Assistant")
st.caption("RAG-powered technical sales assistant for RUBIS 85/95 sliding systems")

# Initialize once
if "retriever" not in st.session_state:
    with st.spinner("Loading retriever..."):
        st.session_state.retriever = MavalRetriever()
    st.success("Retriever ready!")

if "assistant" not in st.session_state:
    with st.spinner("Loading assistant..."):
        st.session_state.assistant = MavalAssistant()
    st.success("Assistant ready!")

# Sidebar filters
st.sidebar.header("Filters")
series_filter = st.sidebar.selectbox(
    "Series filter",
    ["None", "RUBIS 85", "RUBIS 95"],
    index=0,
)
series_value = None if series_filter == "None" else series_filter

debug_mode = st.sidebar.checkbox("Debug mode (show retrieved chunks)", value=False)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"- **{src['id']}** (page {src['page']}, {src['category']})")

# Input
if prompt := st.chat_input("Ask about RUBIS 85/95..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalog..."):
            chunks = st.session_state.retriever.search(prompt, series_filter=series_value)

        if not chunks:
            answer = "I couldn't find relevant information in the catalog for that question."
            sources = []
        else:
            with st.spinner("Generating answer..."):
                result = st.session_state.assistant.answer(prompt, chunks)
                answer = result["answer"]
                sources = result["sources"]

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.write(f"- **{src['id']}** (page {src['page']}, {src['category']})")

        if debug_mode and chunks:
            with st.expander("🔍 Retrieved Chunks (Debug)"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}: `{c['id']}`**")
                    st.text(c["content"][:500] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

# Example questions
st.sidebar.markdown("---")
st.sidebar.subheader("Example Questions")
examples = [
    "What is the inertia of profile 998?",
    "Can I use 24mm glass in RUBIS 85?",
    "Which roller for a 120kg panel?",
    "What profiles for 2-panel RUBIS 95 with dormant 951?",
    "What is the AEV classification?",
]
for ex in examples:
    if st.sidebar.button(ex, key=ex):
        st.session_state.messages.append({"role": "user", "content": ex})
        st.rerun()
