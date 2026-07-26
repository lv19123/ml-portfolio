import os

import requests
import streamlit as st


st.set_page_config(page_title="AI Lecture / Meeting Summarizer", page_icon="AI")

st.title("AI Lecture / Meeting Summarizer")
st.write(
    "Upload lecture, meeting, or study materials now. Processing, summaries, reports, "
    "RAG, and exports will be added in later stages."
)

default_backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
backend_url = st.text_input("Backend URL", value=default_backend_url)
uploaded_file = st.file_uploader(
    "Choose a material",
    type=["pdf", "docx", "txt", "md", "mp3", "wav", "mp4", "mov"],
)
youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")


def reset_material_state(metadata: dict) -> None:
    st.session_state["material_metadata"] = metadata
    st.session_state.pop("extracted_text", None)
    st.session_state["report_content"] = None
    st.session_state["report_type"] = None
    st.session_state["markdown_download"] = None
    st.session_state["pdf_download"] = None
    st.session_state["docx_download"] = None
    st.session_state["topics"] = None
    st.session_state["terms"] = None
    st.session_state["rag_status"] = None
    st.session_state["rag_answer"] = None
    st.session_state["timestamp_segments"] = None

if "report_content" not in st.session_state:
    st.session_state["report_content"] = None
if "report_type" not in st.session_state:
    st.session_state["report_type"] = None
if "markdown_download" not in st.session_state:
    st.session_state["markdown_download"] = None
if "pdf_download" not in st.session_state:
    st.session_state["pdf_download"] = None
if "docx_download" not in st.session_state:
    st.session_state["docx_download"] = None
if "topics" not in st.session_state:
    st.session_state["topics"] = None
if "terms" not in st.session_state:
    st.session_state["terms"] = None
if "rag_status" not in st.session_state:
    st.session_state["rag_status"] = None
if "rag_answer" not in st.session_state:
    st.session_state["rag_answer"] = None
if "timestamp_segments" not in st.session_state:
    st.session_state["timestamp_segments"] = None

if st.button("Upload material", disabled=uploaded_file is None):
    if uploaded_file is not None:
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(
                f"{backend_url.rstrip('/')}/materials/upload",
                files=files,
                timeout=30,
            )
            response.raise_for_status()
            metadata = response.json()
            reset_material_state(metadata)

            st.success("Material uploaded.")
            st.write("material_id:", metadata["material_id"])
            st.write("original filename:", metadata["original_filename"])
            st.write("source type:", metadata["source_type"])
            st.write("status:", metadata["status"])
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

st.subheader("Or process a YouTube lecture")
if st.button("Add YouTube material", disabled=not youtube_url.strip()):
    try:
        response = requests.post(
            f"{backend_url.rstrip('/')}/materials/youtube",
            json={"url": youtube_url},
            timeout=30,
        )
        response.raise_for_status()
        metadata = response.json()
        reset_material_state(metadata)

        st.success("YouTube material added.")
        st.write("material_id:", metadata["material_id"])
        st.write("source type:", metadata["source_type"])
        st.write("source_url:", metadata["source_url"])
        st.write("status:", metadata["status"])
    except requests.RequestException as exc:
        st.error(f"YouTube material creation failed: {exc}")

st.divider()

metadata = st.session_state.get("material_metadata")
if metadata:
    st.subheader("Current material")
    st.write("material_id:", metadata["material_id"])
    st.write("source type:", metadata["source_type"])
    if metadata.get("source_url"):
        st.write("source_url:", metadata["source_url"])
    st.write("status:", metadata["status"])

    if st.button("Process material"):
        try:
            process_response = requests.post(
                f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/process",
                timeout=60,
            )
            process_response.raise_for_status()
            metadata = process_response.json()
            st.session_state["material_metadata"] = metadata
            st.session_state["report_content"] = None
            st.session_state["report_type"] = None
            st.session_state["markdown_download"] = None
            st.session_state["pdf_download"] = None
            st.session_state["docx_download"] = None
            st.session_state["topics"] = None
            st.session_state["terms"] = None
            st.session_state["rag_status"] = None
            st.session_state["rag_answer"] = None
            st.session_state["timestamp_segments"] = None

            text_response = requests.get(
                f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/text",
                timeout=30,
            )
            text_response.raise_for_status()
            st.session_state["extracted_text"] = text_response.json()["text"]

            if metadata.get("has_timestamps"):
                segments_response = requests.get(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/segments",
                    timeout=30,
                )
                segments_response.raise_for_status()
                st.session_state["timestamp_segments"] = segments_response.json()[:10]

            st.success("Material processed.")
        except requests.RequestException as exc:
            st.error(f"Processing failed: {exc}")

    metadata = st.session_state.get("material_metadata", metadata)
    st.write("segments_count:", metadata.get("segments_count"))
    st.write("characters_count:", metadata.get("characters_count"))
    if metadata.get("has_timestamps") is not None:
        st.write("has_timestamps:", metadata.get("has_timestamps"))
    if metadata.get("transcription_engine"):
        st.write("transcription_engine:", metadata.get("transcription_engine"))

    extracted_text = st.session_state.get("extracted_text")
    if extracted_text:
        st.subheader("Extracted text preview")
        st.text_area("Preview", value=extracted_text[:2000], height=300, disabled=True)

    timestamp_segments = st.session_state.get("timestamp_segments")
    if timestamp_segments:
        st.subheader("Timestamp preview")
        st.table(
            [
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                }
                for segment in timestamp_segments
            ]
        )

    is_processed = metadata.get("status") == "processed"
    if is_processed:
        st.subheader("Structure")

        if st.button("Topics and pages/sources"):
            try:
                topics_response = requests.post(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/topics/generate",
                    timeout=90,
                )
                topics_response.raise_for_status()
                st.session_state["topics"] = topics_response.json()["topics"]
                st.success("Topics created.")
            except requests.RequestException as exc:
                st.error(f"Topic extraction failed: {exc}")

        topics = st.session_state.get("topics")
        if topics:
            for topic in topics:
                st.markdown(f"### {topic['title']}")
                st.write(topic["summary"])
                st.caption(f"{topic['source_start']} -> {topic['source_end']}")

        if st.button("Key terms"):
            try:
                terms_response = requests.post(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/terms/generate",
                    timeout=90,
                )
                terms_response.raise_for_status()
                st.session_state["terms"] = terms_response.json()["terms"]
                st.success("Key terms created.")
            except requests.RequestException as exc:
                st.error(f"Key term extraction failed: {exc}")

        terms = st.session_state.get("terms")
        if terms:
            st.table(terms)

        st.subheader("Ask question")

        if st.button("Build RAG index"):
            try:
                rag_response = requests.post(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/rag/build",
                    timeout=60,
                )
                rag_response.raise_for_status()
                st.session_state["rag_status"] = rag_response.json()
                st.success("RAG index created.")
            except requests.RequestException as exc:
                st.error(f"RAG index build failed: {exc}")

        rag_status = st.session_state.get("rag_status")
        if rag_status:
            st.write("retriever:", rag_status.get("retriever"))
            st.write("chunks_count:", rag_status.get("chunks_count"))

        question = st.text_input("Ask a question about the material")
        if st.button("Ask"):
            try:
                ask_response = requests.post(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/ask",
                    json={"question": question, "top_k": 4},
                    timeout=90,
                )
                ask_response.raise_for_status()
                st.session_state["rag_answer"] = ask_response.json()
            except requests.RequestException as exc:
                st.error(f"Question answering failed: {exc}")

        rag_answer = st.session_state.get("rag_answer")
        if rag_answer:
            st.markdown(rag_answer["answer"])
            st.write("Sources")
            st.table(rag_answer["sources"])

        st.subheader("Reports")

        if st.button("Short report"):
            try:
                report_response = requests.post(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/reports/short",
                    timeout=90,
                )
                report_response.raise_for_status()

                content_response = requests.get(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/reports/short",
                    timeout=30,
                )
                content_response.raise_for_status()
                st.session_state["report_type"] = "short"
                st.session_state["report_content"] = content_response.json()["content"]
                st.session_state["markdown_download"] = None
                st.session_state["pdf_download"] = None
                st.session_state["docx_download"] = None
                st.success("Short report created.")
            except requests.RequestException as exc:
                st.error(f"Short report failed: {exc}")

        if st.button("Full cleaned notes"):
            try:
                report_response = requests.post(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/reports/full-clean",
                    timeout=120,
                )
                report_response.raise_for_status()

                content_response = requests.get(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/reports/full-clean",
                    timeout=30,
                )
                content_response.raise_for_status()
                st.session_state["report_type"] = "full-clean"
                st.session_state["report_content"] = content_response.json()["content"]
                st.session_state["markdown_download"] = None
                st.session_state["pdf_download"] = None
                st.session_state["docx_download"] = None
                st.success("Full cleaned notes created.")
            except requests.RequestException as exc:
                st.error(f"Full cleaned notes failed: {exc}")

        if st.session_state.get("report_content"):
            st.markdown(st.session_state["report_content"])

        selected_report_type = st.selectbox(
            "Download report type",
            options=["short", "full-clean"],
        )
        base_filename = (
            "short_report" if selected_report_type == "short" else "full_clean_notes"
        )

        if st.button("Download Markdown"):
            try:
                download_response = requests.get(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/download/md",
                    params={"report_type": selected_report_type},
                    timeout=30,
                )
                download_response.raise_for_status()
                st.session_state["markdown_download"] = {
                    "content": download_response.text,
                    "filename": f"{base_filename}.md",
                }
            except requests.RequestException as exc:
                st.error(f"Markdown download failed: {exc}")

        markdown_download = st.session_state.get("markdown_download")
        if markdown_download:
            st.download_button(
                "Save Markdown file",
                data=markdown_download["content"],
                file_name=markdown_download["filename"],
                mime="text/markdown",
            )

        if st.button("Download PDF"):
            try:
                download_response = requests.get(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/download/pdf",
                    params={"report_type": selected_report_type},
                    timeout=60,
                )
                download_response.raise_for_status()
                st.session_state["pdf_download"] = {
                    "content": download_response.content,
                    "filename": f"{base_filename}.pdf",
                }
            except requests.RequestException as exc:
                st.error(f"PDF download failed: {exc}")

        pdf_download = st.session_state.get("pdf_download")
        if pdf_download:
            st.download_button(
                "Save PDF file",
                data=pdf_download["content"],
                file_name=pdf_download["filename"],
                mime="application/pdf",
            )

        if st.button("Download DOCX"):
            try:
                download_response = requests.get(
                    f"{backend_url.rstrip('/')}/materials/{metadata['material_id']}/download/docx",
                    params={"report_type": selected_report_type},
                    timeout=60,
                )
                download_response.raise_for_status()
                st.session_state["docx_download"] = {
                    "content": download_response.content,
                    "filename": f"{base_filename}.docx",
                }
            except requests.RequestException as exc:
                st.error(f"DOCX download failed: {exc}")

        docx_download = st.session_state.get("docx_download")
        if docx_download:
            st.download_button(
                "Save DOCX file",
                data=docx_download["content"],
                file_name=docx_download["filename"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    else:
        st.info("Process the material before generating reports.")
else:
    st.info("Upload a material before processing.")

st.info("Long media files may take time to process. YouTube support uses available transcripts only.")
