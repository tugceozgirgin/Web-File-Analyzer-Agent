from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import List, Optional

from src.agents.state import Filters


class StructureAnalyzerPrompts:

    @staticmethod
    def build_structure_analyzer_prompt(
        filters: Filters,
        human_feedback: str | None = None,
    ) -> List[BaseMessage]:
        """Prompt for the first run — instructs the LLM to call firecrawl_scrape."""
        messages: List[BaseMessage] = []

        filter_parts: List[str] = []
        if filters.file_type:
            filter_parts.append(
                f"- File types: {', '.join(ft.value for ft in filters.file_type)}"
            )
        if filters.categories:
            filter_parts.append(
                f"- Categories: {', '.join(filters.categories)}"
            )
        if filters.start_date:
            filter_parts.append(
                f"- Start date: {filters.start_date.strftime('%Y-%m-%d')}"
            )
        if filters.end_date:
            filter_parts.append(
                f"- End date: {filters.end_date.strftime('%Y-%m-%d')}"
            )

        filters_text = (
            "\n".join(filter_parts)
            if filter_parts
            else "No additional filters (return all files found)."
        )

        system_message = SystemMessage(content="""
You are a Structure Analyzer assistant for a Web File Analyzer.
You have access to ONE tool:

TOOL: firecrawl_scrape(url, formats, onlyMainContent, excludeTags, waitFor, ...)
- Returns: The web page content converted to clean markdown.
- USE FOR: Retrieving the content of the target URL so you can analyze it.
- CRITICAL: You MUST pass these exact parameters to reduce content size:
  * formats: ["markdown"] (required - returns clean markdown, not full HTML)
  * onlyMainContent: true (required - excludes headers, footers, sidebars)
  * excludeTags: ["nav", "header", "footer", "aside", "script", "style", "noscript"] (optional but recommended - filters out navigation and boilerplate)
  * waitFor: 30000 (required - wait up to 30 seconds for dynamic content to load)

WORKFLOW:
1. Call firecrawl_scrape with the URL provided by the user. IMPORTANT: Always include
   formats=["markdown"], onlyMainContent=true, and excludeTags=["nav", "header", "footer", "aside"]
   to get only the main content without navigation and boilerplate HTML.
2. Extract the page title from the page content (look for <title> tags, h1 headings, or page headers).
3. Write a short one-sentence description of what the page contains (maximum 1 sentence).
4. Once you receive the page content from the tool, analyze it to extract every
   downloadable file link. Supported file types are: pdf, docx, xlsx, csv.
5. Group the files under the section heading or title they appear under on the
   page. Each group becomes one item in "file_structures" with a "title" field.
   If a file does not belong to any clear section, use a sensible default title
   such as "Other Files".
6. For each file extract:
   - file_name: the human-readable name or anchor text of the link.
   - file_type: one of pdf, docx, xlsx, csv (inferred from the URL extension
     or surrounding context).
   - file_url: the full, absolute download URL. If the URL on the page is
     relative, combine it with the base URL to form an absolute URL.
7. Apply the provided filters to narrow down the results:
   - file_type filter: only include files whose type matches one of the
     requested types.
   - categories filter: only include files whose title or name is relevant
     to at least one of the given category keywords.
   - start_date / end_date filter: only include files whose associated date
     (parsed from the file name, title, or surrounding text) falls within the
     given date range (inclusive).
8. If no files match after filtering, return an empty file_structures list.
9. Do NOT fabricate file links. Only return files actually present in the page.

IMPORTANT:
- You MUST call firecrawl_scrape first before producing the output.
- After analyzing, respond with ONLY valid JSON — no extra text.

OUTPUT FORMAT (valid JSON):
{
  "page_title": "Page Title Here",
  "page_description": "A short one-sentence description of what the page contains.",
  "file_structures": [
    {
      "title": "Section or group title",
      "file_names": [
        {
          "file_name": "Human-readable file name",
          "file_type": "pdf",
          "file_url": "https://example.com/path/to/file.pdf"
        }
      ]
    }
  ]
}
""".strip())

        human_content = f"""Fetch and analyze this URL: {filters.url}

Active Filters:
{filters_text}"""

        if human_feedback:
            human_content += (
                f"\n\nIMPORTANT — User Feedback (from a previous rejected analysis):\n"
                f"{human_feedback}\n"
                f"Take this feedback into account when selecting which files to include "
                f"or exclude from the results."
            )

        human_message = HumanMessage(content=human_content.strip())

        messages.append(system_message)
        messages.append(human_message)

        return messages

    @staticmethod
    def build_reanalysis_prompt(
        filters: Filters,
        page_content: str,
        human_feedback: Optional[str] = None,
    ) -> List[BaseMessage]:
        """Prompt for retry / cache-hit — page content is already available,
        so the LLM re-analyses WITHOUT calling any tool.

        ``human_feedback`` is optional; when *None* (cache-hit path) the
        feedback section is simply omitted from the prompt.
        """
        messages: List[BaseMessage] = []

        filter_parts: List[str] = []
        if filters.file_type:
            filter_parts.append(
                f"- File types: {', '.join(ft.value for ft in filters.file_type)}"
            )
        if filters.categories:
            filter_parts.append(
                f"- Categories: {', '.join(filters.categories)}"
            )
        if filters.start_date:
            filter_parts.append(
                f"- Start date: {filters.start_date.strftime('%Y-%m-%d')}"
            )
        if filters.end_date:
            filter_parts.append(
                f"- End date: {filters.end_date.strftime('%Y-%m-%d')}"
            )

        filters_text = (
            "\n".join(filter_parts)
            if filter_parts
            else "No additional filters (return all files found)."
        )

        workflow_step_6 = (
            "6. Apply the provided filters AND the user feedback to narrow down the results."
            if human_feedback
            else "6. Apply the provided filters to narrow down the results."
        )

        system_message = SystemMessage(content=f"""
You are a Structure Analyzer assistant for a Web File Analyzer.

The page has already been scraped and the content is provided below.
Do NOT call any tools. Just re-analyze the provided content.

WORKFLOW:
1. Extract the page title from the page content (look for headings or page headers).
2. Write a short one-sentence description of what the page contains (maximum 1 sentence).
3. Analyze the page content to extract every downloadable file link.
   Supported file types are: pdf, docx, xlsx, csv.
4. Group the files under the section heading or title they appear under on the
   page. Each group becomes one item in "file_structures" with a "title" field.
   If a file does not belong to any clear section, use a sensible default title
   such as "Other Files".
5. For each file extract:
   - file_name: the human-readable name or anchor text of the link.
   - file_type: one of pdf, docx, xlsx, csv (inferred from the URL extension
     or surrounding context).
   - file_url: the full, absolute download URL.
{workflow_step_6}
7. If no files match after filtering, return an empty file_structures list.
8. Do NOT fabricate file links. Only return files actually present in the content.

IMPORTANT:
- Do NOT call any tools. The page content is already provided.
- After analyzing, respond with ONLY valid JSON — no extra text.

OUTPUT FORMAT (valid JSON):
{{
  "page_title": "Page Title Here",
  "page_description": "A short one-sentence description of what the page contains.",
  "file_structures": [
    {{
      "title": "Section or group title",
      "file_names": [
        {{
          "file_name": "Human-readable file name",
          "file_type": "pdf",
          "file_url": "https://example.com/path/to/file.pdf"
        }}
      ]
    }}
  ]
}}
""".strip())

        human_parts = [
            f"URL: {filters.url}",
            "",
            f"Active Filters:\n{filters_text}",
        ]
        if human_feedback:
            human_parts.append(
                f"\nUser Feedback (MUST be applied):\n{human_feedback}"
            )
        human_parts.append(
            f"\nPage Content (already scraped — do NOT call any tools):\n{page_content}"
        )

        human_message = HumanMessage(content="\n".join(human_parts).strip())

        messages.append(system_message)
        messages.append(human_message)

        return messages
