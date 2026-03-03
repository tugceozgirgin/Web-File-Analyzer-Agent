import asyncio
import logging
import time
from typing import List, Optional, Dict

from src.agents import BaseAgent
from src.agents.state import (
    WebFileAnalyzerState,
    FileStructure,
    File,
    FileSummary,
    FileTypes,
    Filters,
)
from src.agents.prompts import FileReaderPrompts
from src.models import Models

from src.agents.tools.pdf_reader_tool import pdf_reader_tool
from src.agents.tools.docx_reader_tool import docx_reader_tool
from src.agents.tools.excel_reader_tool import excel_reader_tool
from src.agents.tools.csv_reader_tool import csv_reader_tool

logger = logging.getLogger(__name__)

_LLM_MAX_RETRIES = 5
_LLM_RETRY_BASE_DELAY = 1.0
_SEMAPHORE_LIMIT = 2

_READER_DISPATCH = {
    FileTypes.PDF: pdf_reader_tool,
    FileTypes.DOCX: docx_reader_tool,
    FileTypes.XLSX: excel_reader_tool,
    FileTypes.CSV: csv_reader_tool,
}


class FileReaderAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="File Reader Agent", tools=[], model_name="gpt-4o-mini")
        self.llm = Models.get_openai_model(model_name=self.model_name)
        self._faiss_store = None

    @staticmethod
    def _flatten_files(file_structures: List[FileStructure]) -> List[File]:
        files: List[File] = []
        for fs in file_structures:
            if fs.file_names:
                files.extend(fs.file_names)
        return files

    def _process_single_file_sync(self, file: File) -> FileSummary:
        """Download, extract text, and LLM-summarise a single file (blocking).

        Returns a cached ``FileSummary`` immediately when available.
        """
        from src.cache import get_cache_manager

        cache = get_cache_manager()

        cached = cache.get_file_summary(file.file_url)
        if cached is not None:
            logger.info("Using cached summary for %s", file.file_name)
            return cached

        reader = _READER_DISPATCH.get(file.file_type)
        if reader is None:
            logger.warning(
                "No reader for file type %s — skipping %s",
                file.file_type,
                file.file_name,
            )
            return FileSummary(
                file_name=file.file_name,
                file_type=file.file_type,
                file_url=file.file_url,
                error=(
                    f"Unsupported file type '{file.file_type.value}'. "
                    f"Only PDF, DOCX, XLSX, and CSV files are supported."
                ),
            )

        try:
            extracted_content: str = reader.invoke({"file_url": file.file_url})
        except Exception as exc:
            logger.error("Failed to read %s: %s", file.file_name, exc)
            return FileSummary(
                file_name=file.file_name,
                file_type=file.file_type,
                file_url=file.file_url,
                error=f"Could not download or read this file: {exc}",
            )

        messages = FileReaderPrompts.build_file_summary_prompt(
            file_name=file.file_name,
            file_type=file.file_type.value,
            extracted_content=extracted_content,
        )

        summary = None
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                response = self.llm.invoke(messages)
                summary = response.content.strip()
                break
            except Exception as exc:
                err_str = str(exc)
                if "rate_limit" in err_str or "Rate limit" in err_str or "429" in err_str:
                    wait = _LLM_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Rate-limited on %s (attempt %d/%d) — waiting %.1fs",
                        file.file_name, attempt + 1, _LLM_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM summarisation failed for %s: %s", file.file_name, exc)
                    break

        if summary is None:
            summary = (
                extracted_content[:200] + "…"
                if len(extracted_content) > 200
                else extracted_content
            )

        result = FileSummary(
            file_name=file.file_name,
            file_type=file.file_type,
            file_url=file.file_url,
            summary=summary,
        )

        # cache only successful results (no error)
        if not result.error:
            cache.set_file_summary(result)

        return result

    async def _process_single_file(self, file: File, semaphore: asyncio.Semaphore) -> FileSummary:
        """Semaphore-guarded wrapper that runs the blocking work in a thread."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._process_single_file_sync, file
            )

    def _store_in_faiss(self, summaries: List[FileSummary]) -> None:
        """Optionally store summaries in a FAISS vector store for later
        retrieval.  Silently skipped if dependencies are missing."""
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
        except ImportError:
            logger.warning(
                "FAISS or OpenAIEmbeddings not available — skipping vector store"
            )
            return

        if not summaries:
            return

        documents = [
            Document(
                page_content=s.summary,
                metadata={
                    "file_name": s.file_name,
                    "file_type": s.file_type.value,
                    "file_url": s.file_url,
                },
            )
            for s in summaries
        ]

        try:
            embeddings = OpenAIEmbeddings()
            self._faiss_store = FAISS.from_documents(documents, embeddings)
            logger.info(
                "Stored %d file summaries in FAISS vector store", len(documents)
            )
        except Exception as exc:
            logger.warning("Failed to build FAISS store: %s", exc)

    def similarity_search(self, query: str, k: int = 4):
        """Run a similarity search against the stored file summaries.

        Returns an empty list if the FAISS store has not been built yet.
        """
        if self._faiss_store is None:
            return []
        return self._faiss_store.similarity_search(query, k=k)

    @staticmethod
    def _format_output(
        page_title: Optional[str],
        page_description: Optional[str],
        filters: Optional[Filters],
        file_structures: List[FileStructure],
        summaries_index: Dict[str, str],
        errors_index: Dict[str, str],
    ) -> str:
        """Build the final markdown output with per-file summaries inlined."""
        lines: List[str] = []

        lines.append(f"**Title:** {page_title or '(Not available)'}")
        lines.append(f"**Description:** {page_description or '(Not available)'}")

        filter_parts: List[str] = []
        if filters:
            if filters.start_date and filters.end_date:
                start_str = filters.start_date.strftime("%Y-%m-%d")
                end_str = filters.end_date.strftime("%Y-%m-%d")
                filter_parts.append(f"Date Period: {start_str} - {end_str}")
            elif filters.start_date:
                filter_parts.append(
                    f"Start Date: {filters.start_date.strftime('%Y-%m-%d')}"
                )
            elif filters.end_date:
                filter_parts.append(
                    f"End Date: {filters.end_date.strftime('%Y-%m-%d')}"
                )
            if filters.categories:
                filter_parts.append(f"Categories: {', '.join(filters.categories)}")
            if filters.file_type:
                filter_parts.append(
                    f"File format: {', '.join(ft.value for ft in filters.file_type)}"
                )

        if filter_parts:
            lines.append("")
            lines.append("**Applied Filters:**")
            for part in filter_parts:
                lines.append(f"   {part}")

        lines.append("")
        lines.append("**File Structure:**")
        if file_structures:
            for fs in file_structures:
                if fs.title:
                    lines.append(f"- **{fs.title}**")
                if fs.file_names:
                    for file in fs.file_names:
                        lines.append(
                            f"  - {file.file_name} ({file.file_type.value}) "
                            f"- {file.file_url}"
                        )
                        error = errors_index.get(file.file_url)
                        if error:
                            lines.append(f"    - ⚠️ *{error}*")
                        else:
                            summary = summaries_index.get(file.file_url, "")
                            if summary:
                                lines.append(f"    - *Summary: {summary}*")
        else:
            lines.append("- No files found matching the criteria.")

        return "\n".join(lines)

    async def _process_all_files(self, files: List[File]) -> List[FileSummary]:
        """Process files with at most ``_SEMAPHORE_LIMIT`` running concurrently."""
        semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)

        async def _safe_process(file: File) -> FileSummary:
            try:
                return await self._process_single_file(file, semaphore)
            except Exception as exc:
                logger.error("Unexpected error processing %s: %s", file.file_name, exc)
                return FileSummary(
                    file_name=file.file_name,
                    file_type=file.file_type,
                    file_url=file.file_url,
                    error=f"Processing failed: {exc}",
                )

        tasks = [_safe_process(f) for f in files]
        return list(await asyncio.gather(*tasks))

    def forward(self, state: WebFileAnalyzerState) -> WebFileAnalyzerState:
        file_structures: List[FileStructure] = state.get("file_structures", [])
        files = self._flatten_files(file_structures)

        if not files:
            state["output"] = "No files to process."
            return state

        logger.info("FileReaderAgent processing %d file(s)", len(files))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                summaries = pool.submit(
                    asyncio.run, self._process_all_files(files)
                ).result()
        else:
            summaries = asyncio.run(self._process_all_files(files))

        state["file_summaries"] = summaries

        self._store_in_faiss(summaries)

        summaries_index = {s.file_url: s.summary for s in summaries if s.summary}
        errors_index = {s.file_url: s.error for s in summaries if s.error}
        state["output"] = self._format_output(
            page_title=state.get("page_title"),
            page_description=state.get("page_description"),
            filters=state.get("filters"),
            file_structures=file_structures,
            summaries_index=summaries_index,
            errors_index=errors_index,
        )

        logger.info(
            "FileReaderAgent completed — %d summaries produced", len(summaries)
        )
        return state
