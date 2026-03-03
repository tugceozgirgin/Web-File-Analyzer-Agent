from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import List


class FileReaderPrompts:

    @staticmethod
    def build_file_summary_prompt(
        file_name: str,
        file_type: str,
        extracted_content: str,
    ) -> List[BaseMessage]:
        """Return a [SystemMessage, HumanMessage] pair that instructs the LLM
        to produce a concise 2-3 sentence summary of the given file content."""

        system_msg = SystemMessage(
            content=(
                "You are a document summarisation assistant. "
                "Given the name, type, and extracted content of a file, "
                "produce a **1-2 sentence** summary — be as brief as possible. "
                "Focus only on the single most important takeaway. "
                "Do NOT repeat the file name or type. "
                "For tabular data, just state the key dimensions (sheets, rows, columns)."
            )
        )

        human_msg = HumanMessage(
            content=(
                f"**File name:** {file_name}\n"
                f"**File type:** {file_type}\n\n"
                f"**Extracted content:**\n{extracted_content}"
            )
        )

        return [system_msg, human_msg]
