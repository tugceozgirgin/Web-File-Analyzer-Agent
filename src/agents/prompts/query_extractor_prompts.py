from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from typing import List


class QueryExtractorPrompts:

    @staticmethod
    def build_query_extractor_prompt(
        user_input: str,
        human_feedback: str | None = None,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = []

        system_message = SystemMessage(content="""
You are a Query Extraction assistant for a Web File Analyzer.

Your ONLY task is to read the user's query and extract a set of filters describing
which files should be searched and downloaded from a given website.

You MUST respond strictly as a structured `Filters` object with the following fields:
- url: the exact website URL the user explicitly mentions.
- file_type: a list of file types from {pdf, docx, xlsx, csv} mentioned in the query (or null if none).
- categories: a list of topical or domain keywords that describe the content the user wants if he mentions it. If he does not mention it, then set it to null.
- start_date: the earliest date of the files requested, or null if not specified.
- end_date: the latest date of the files requested, or null if not specified.

Date handling guidelines:
- If the user asks for reports for a single year (e.g. "2025 reports"), set
  start_date to the first day of that year and end_date to the last day of that year.
- If the user provides a date range, map it to start_date and end_date.
- If no clear date information is given, leave both start_date and end_date as null.
- Do not go to url just extract the filters from the query. Or made up any category filters if user does not specifically mention it.

Irrelevant query handling:
- If the query is NOT about finding/downloading files from a website, or there is
  no URL in the query, consider it irrelevant.
- For irrelevant queries, set ALL fields (url, file_type, categories, start_date, end_date) to null.

If there is USER FEEDBACK from a previous analysis attempt, take it into account
when extracting filters. For example the user may want to narrow the date range,
change file types, or adjust categories based on what they saw.
""".strip())

        human_content = f"User Query:\n{user_input}"
        if human_feedback:
            human_content += (
                f"\n\nUser Feedback (from a previous rejected analysis):\n"
                f"{human_feedback}"
            )

        human_message = HumanMessage(content=human_content.strip())

        messages.append(system_message)
        messages.append(human_message)

        return messages