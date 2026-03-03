import logging
import json
import re
from typing import Optional

from langchain_core.messages import SystemMessage, ToolMessage
from pydantic import ValidationError

from src.agents import BaseAgent
from src.agents.state import WebFileAnalyzerState, FileStructureList, Filters
from src.agents.prompts import StructureAnalyzerPrompts
from src.models import Models

logger = logging.getLogger(__name__)


class StructureAnalyzerAgent(BaseAgent):
    def __init__(self, tools: list):
        super().__init__(name="Structure Analyzer Agent", tools=tools, model_name="gpt-4o-mini")
        self.llm = Models.get_openai_model(model_name=self.model_name)

        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    @staticmethod
    def parse_file_structures(content: str) -> FileStructureList:
        """Parse the LLM's text response into a FileStructureList."""
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON: %s", content[:200])
            return FileStructureList()

        try:
            return FileStructureList.model_validate(data)
        except ValidationError as exc:
            logger.warning("Validation error when parsing file structures: %s", exc)
            return FileStructureList()

    @staticmethod
    def format_file_structure_summary(
        page_title: Optional[str],
        page_description: Optional[str],
        filters: Optional[Filters],
        file_structures: list,
    ) -> str:
        """Format the file structure summary for display to the user."""
        lines = []

        if page_title:
            lines.append(f"**Title:** {page_title}")
        else:
            lines.append("**Title:** (Not available)")

        if page_description:
            lines.append(f"**Description:** {page_description}")
        else:
            lines.append("**Description:** (Not available)")

        filter_parts = []
        if filters:
            if filters.start_date and filters.end_date:
                start_str = filters.start_date.strftime("%Y-%m-%d")
                end_str = filters.end_date.strftime("%Y-%m-%d")
                filter_parts.append(f"Date Period: {start_str} - {end_str}")
            elif filters.start_date:
                filter_parts.append(f"Start Date: {filters.start_date.strftime('%Y-%m-%d')}")
            elif filters.end_date:
                filter_parts.append(f"End Date: {filters.end_date.strftime('%Y-%m-%d')}")

            if filters.categories:
                filter_parts.append(f"Categories: {', '.join(filters.categories)}")

            if filters.file_type:
                filter_parts.append(f"File format: {', '.join(ft.value for ft in filters.file_type)}")

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
                        lines.append(f"  - {file.file_name} ({file.file_type}) - {file.file_url}")
        else:
            lines.append("- No files found matching the criteria.")

        return "\n".join(lines)

    def _save_result(self, state, filters, file_structure_list):
        """Persist parsed results + formatted summary into state."""
        state["file_structures"] = file_structure_list.file_structures
        state["page_title"] = file_structure_list.page_title
        state["page_description"] = file_structure_list.page_description
        state["file_structure_summary"] = self.format_file_structure_summary(
            page_title=file_structure_list.page_title,
            page_description=file_structure_list.page_description,
            filters=filters,
            file_structures=file_structure_list.file_structures,
        )

    @staticmethod
    def _has_no_files(file_structure_list: FileStructureList) -> bool:
        """Return True when the parsed result contains zero files."""
        return sum(
            len(fs.file_names or []) for fs in file_structure_list.file_structures
        ) == 0

    def forward(self, state: WebFileAnalyzerState) -> WebFileAnalyzerState:
        from src.cache import get_cache_manager

        filters = state.get("filters")

        if filters is None or getattr(filters, "url", None) is None:
            state["file_structures"] = []
            return state

        cache = get_cache_manager()
        cached_content = state.get("page_content")
        human_feedback = state.get("human_feedback")
        state_messages = list(state.get("messages", []))
        in_tool_loop = state_messages and isinstance(state_messages[-1], ToolMessage)


        if not cached_content and not in_tool_loop:
            cached_page = cache.get_page(filters.url)
            if cached_page:
                cached_content = cached_page
                state["page_content"] = cached_content

        try:
            if cached_content and not in_tool_loop:
                logger.info(
                    "Analysing cached page content%s",
                    " with user feedback" if human_feedback else "",
                )
                prompt = StructureAnalyzerPrompts.build_reanalysis_prompt(
                    filters=filters,
                    page_content=cached_content,
                    human_feedback=human_feedback,
                )
                response = self.llm.invoke(prompt)
                file_structure_list = self.parse_file_structures(response.content)

                if self._has_no_files(file_structure_list):
                    state["file_structures"] = []
                    state["output"] = (
                        "ℹ️ I was able to access the page, but no downloadable files "
                        "(PDF, DOCX, XLSX, CSV) were found matching your criteria.\n\n"
                        "You might want to check the URL or adjust your filters."
                    )
                    state["messages"] = [response]
                    return state

                self._save_result(state, filters, file_structure_list)
                state["messages"] = [response]
                return state

            if in_tool_loop:
                tool_msg = state_messages[-1]

                if getattr(tool_msg, "status", None) == "error":
                    logger.warning("Firecrawl tool returned an error: %s", str(tool_msg.content)[:300])
                    state["file_structures"] = []
                    state["output"] = (
                        f"⚠️ I was unable to reach the URL:\n\n"
                        f"**{filters.url}**\n\n"
                        f"The page could not be loaded. Please make sure the URL is "
                        f"correct and accessible, then try again."
                    )
                    return state

                if hasattr(tool_msg, "content") and tool_msg.content:
                    page_content = (
                        tool_msg.content
                        if isinstance(tool_msg.content, str)
                        else str(tool_msg.content)
                    )
                    state["page_content"] = page_content
                    cache.set_page(filters.url, page_content)

                prompt_messages = StructureAnalyzerPrompts.build_structure_analyzer_prompt(
                    filters=filters,
                    human_feedback=human_feedback,
                )
                messages = [prompt_messages[0], prompt_messages[1]] + state_messages

            else:
                prompt_messages = StructureAnalyzerPrompts.build_structure_analyzer_prompt(
                    filters=filters,
                    human_feedback=human_feedback,
                )
                messages = [prompt_messages[0], prompt_messages[1]]

            response = self.llm_with_tools.invoke(messages)

            new_messages = [response]

            if hasattr(response, "tool_calls") and response.tool_calls:
                pass
            else:
                file_structure_list = self.parse_file_structures(response.content)

                if self._has_no_files(file_structure_list):
                    state["file_structures"] = []
                    state["output"] = (
                        "ℹ️ I was able to access the page, but no downloadable files "
                        "(PDF, DOCX, XLSX, CSV) were found matching your criteria.\n\n"
                        "You might want to check the URL or adjust your filters."
                    )
                    state["messages"] = new_messages
                    return state

                self._save_result(state, filters, file_structure_list)

            state["messages"] = new_messages
            return state

        except Exception as exc:
            logger.error("Error in StructureAnalyzerAgent: %s", exc)
            state["file_structures"] = []
            state["output"] = (
                "⚠️ Something went wrong while analyzing the page.\n\n"
                f"**Error:** {exc}\n\n"
                "Please try again or check that the URL is correct."
            )
            return state
