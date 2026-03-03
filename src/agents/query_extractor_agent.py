from typing import Dict, Any
from src.agents import BaseAgent
from src.agents.state import WebFileAnalyzerState
from src.models import Models
from src.agents.prompts import QueryExtractorPrompts
from src.agents.state import Filters

class QueryExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Query Extractor Agent", tools=[], model_name="gpt-5-nano")
        self.llm = Models.get_openai_model(model_name=self.model_name).with_structured_output(Filters)
        

    def forward(self, state: WebFileAnalyzerState) -> WebFileAnalyzerState:
        prompt = QueryExtractorPrompts.build_query_extractor_prompt(
            user_input=state["input"],
            human_feedback=state.get("human_feedback"),
        )
        filters: Filters = self.llm.invoke(prompt)

        state["filters"] = filters

        if (
            getattr(filters, "url", None) is None
            and getattr(filters, "file_type", None) in (None, [])
            and getattr(filters, "categories", None) in (None, [])
            and getattr(filters, "start_date", None) is None
            and getattr(filters, "end_date", None) is None
        ):
            state["output"] = (
                "Your question does not look related to extracting files from a given URL. "
                "Please ask a question about finding or filtering files (such as PDF, CSV, reports) "
                "from a specific website address."
            )

        return state