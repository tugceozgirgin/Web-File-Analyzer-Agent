import asyncio
import queue
import threading

import streamlit as st
import os
from agents.graph import WebFileAnalyzerGraph
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="Web File Analyzer Agent",
    page_icon="📂",
    layout="wide",
)

if "graph" not in st.session_state:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    st.session_state.graph = WebFileAnalyzerGraph(model_name=model_name)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_review" not in st.session_state:
    st.session_state.awaiting_review = False
if "thread_config" not in st.session_state:
    st.session_state.thread_config = None
if "pending_summary" not in st.session_state:
    st.session_state.pending_summary = None
if "show_feedback_form" not in st.session_state:
    st.session_state.show_feedback_form = False

st.title("📂 Web File Analyzer Agent")
st.markdown(
    "Paste a URL and I'll scan the page for downloadable files, "
    "extract their content, and generate a short summary for each one."
)

with st.expander("🔍 Available filters you can include in your query", expanded=False):
    st.markdown(
        """
| Filter | Example query |
|---|---|
| **Time period** | *"only files from 2024-2025"* |
| **Categories** | *"files about Automotive industry"*, *"Retail reports"* |
| **File type** | *"only PDFs"*, *"xlsx and csv files"* |

**Supported formats:** PDF · DOCX · XLSX · CSV

Simply mention any combination of these in your message and the agent will apply them automatically.
"""
    )

with st.sidebar:
    st.header("⚡ Cache")

    from cache import get_cache_manager

    _cache = get_cache_manager()
    _stats = _cache.get_stats()

    st.markdown("**Stored**")
    c1, c2 = st.columns(2)
    c1.metric("📄 Pages", _stats.get("cached_pages", 0))
    c2.metric("📁 Files", _stats.get("cached_files", 0))

    st.markdown("**Session hits / misses**")
    c3, c4 = st.columns(2)
    c3.metric("📄 Page hits", _stats.get("page_hits", 0), delta=f"{_stats.get('page_misses', 0)} misses")
    c4.metric("📁 File hits", _stats.get("file_hits", 0), delta=f"{_stats.get('file_misses', 0)} misses")

    if st.button("🗑️ Clear All Caches"):
        _cache.clear_all()
        st.success("All caches cleared!")
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def _run_graph_stream(graph_instance, input_data, thread_config):
    """Run ``compiled_graph.astream`` in a worker thread.

    Returns (progress_chunks, final_state_values).
    """
    q: queue.Queue = queue.Queue()
    result = {"state": None, "error": None}

    async def _run():
        try:
            compiled = graph_instance.compiled_graph
            current_node = ""
            async for event in compiled.astream(
                input_data, thread_config, stream_mode="updates"
            ):
                for node_name, _node_output in event.items():
                    if node_name != current_node:
                        current_node = node_name
                        q.put(f"\n🔄 *{node_name}*\n")
            result["state"] = compiled.get_state(thread_config).values
        except Exception as exc:
            result["error"] = exc
        finally:
            q.put(None)

    def _thread_target():
        asyncio.run(_run())

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()

    chunks: list[str] = []
    while True:
        chunk = q.get()
        if chunk is None:
            break
        chunks.append(chunk)

    t.join()
    if result["error"]:
        raise result["error"]
    return chunks, result["state"]


if st.session_state.awaiting_review:
    with st.container():
        st.info("📋 Please review the file structure analysis:")
        st.markdown(st.session_state.pending_summary)

        if not st.session_state.show_feedback_form:
            col1, col2 = st.columns(2)

            if col1.button("✅ Accept", type="primary", key="accept_btn"):
                graph = st.session_state.graph
                config = st.session_state.thread_config

                graph.update_state(
                    config,
                    {"human_approval": "accept"},
                    as_node="human_review",
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": st.session_state.pending_summary}
                )
                st.session_state.messages.append(
                    {"role": "user", "content": "✅ Accepted"}
                )

                with st.spinner("Processing files..."):
                    _chunks, final_state = _run_graph_stream(graph, None, config)

                if final_state and final_state.get("output"):
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_state["output"]}
                    )

                st.session_state.awaiting_review = False
                st.session_state.pending_summary = None
                st.rerun()

            if col2.button("❌ Reject", key="reject_btn"):
                st.session_state.show_feedback_form = True
                st.rerun()

        else:
            feedback = st.text_input(
                "What should be different?",
                placeholder="e.g. only include 2024 reports",
                key="feedback_input",
            )
            if st.button("Submit Feedback", key="submit_feedback_btn"):
                graph = st.session_state.graph
                config = st.session_state.thread_config
                feedback_text = feedback.strip() if feedback else "Please try again."

                graph.update_state(
                    config,
                    {
                        "human_approval": "reject",
                        "human_feedback": feedback_text,
                        "messages": [
                            HumanMessage(
                                content=(
                                    f"User rejected the analysis. "
                                    f"Feedback: {feedback_text}"
                                )
                            )
                        ],
                    },
                    as_node="human_review",
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": st.session_state.pending_summary}
                )
                st.session_state.messages.append(
                    {"role": "user", "content": f"❌ Rejected — {feedback_text}"}
                )

                with st.spinner("Re-analyzing..."):
                    _chunks, final_state = _run_graph_stream(graph, None, config)

                if final_state and final_state.get("file_structure_summary"):
                    st.session_state.pending_summary = (
                        final_state["file_structure_summary"]
                    )
                    st.session_state.awaiting_review = True
                else:
                    st.session_state.awaiting_review = False
                    st.session_state.pending_summary = None

                st.session_state.show_feedback_form = False
                st.rerun()


else:
    if prompt := st.chat_input("Enter a URL to analyze..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                graph = st.session_state.graph
                thread_config = graph.new_thread_config()
                st.session_state.thread_config = thread_config

                response_placeholder = st.empty()
                progress = ""

                with st.spinner("Analyzing..."):
                    chunks, final_state = _run_graph_stream(
                        graph, {"input": prompt}, thread_config
                    )
                    for chunk in chunks:
                        progress += chunk
                        response_placeholder.markdown(progress + "▌")

                if final_state and final_state.get("file_structure_summary"):
                    summary = final_state["file_structure_summary"]
                    response_placeholder.markdown(summary)
                    st.session_state.pending_summary = summary
                    st.session_state.awaiting_review = True
                    st.rerun()
                else:
                    output = (final_state or {}).get("output", "") or progress
                    response_placeholder.markdown(output)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": output}
                    )

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
