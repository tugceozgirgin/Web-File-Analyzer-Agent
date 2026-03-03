import asyncio
import concurrent.futures
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from src.agents.state import WebFileAnalyzerState
from src.agents import QueryExtractorAgent, StructureAnalyzerAgent, FileReaderAgent
from src.agents.tools.fetch_page_tool import get_firecrawl_tools


class WebFileAnalyzerGraph:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tools = get_firecrawl_tools()
        self.memory = MemorySaver()
        self._compiled = None

    @property
    def compiled_graph(self):
        """Lazily build and cache the compiled graph (with checkpointer)."""
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled

    @staticmethod
    def _should_analyze(state: WebFileAnalyzerState):
        """After query_extractor: proceed to structure_analyzer if a URL
        was extracted, otherwise go straight to END."""
        filters = state.get("filters")
        if filters and getattr(filters, "url", None):
            return "structure_analyzer"
        return END

    @staticmethod
    def _route_after_structure_analyzer(state: WebFileAnalyzerState):
        """After structure_analyzer: route to tools if there are pending
        tool calls, otherwise send to human_review for approval.
        If an error or no-files output was set, skip straight to END."""
        if state.get("output"):
            return END
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if (
                isinstance(last, AIMessage)
                and hasattr(last, "tool_calls")
                and last.tool_calls
            ):
                return "tools"
        return "human_review"

    @staticmethod
    def _route_after_human_review(state: WebFileAnalyzerState):
        """After human_review: route based on user approval."""
        approval = state.get("human_approval")
        if approval == "accept":
            return "file_reader"
        elif approval == "reject":
            return "query_extractor"
        return END


    def _build(self):
        graph = StateGraph(WebFileAnalyzerState)

        query_extractor = QueryExtractorAgent()
        structure_analyzer = StructureAnalyzerAgent(tools=self.tools)
        file_reader = FileReaderAgent()

        def human_review(state: WebFileAnalyzerState):
            """No-op node — the graph interrupts *before* this node so
            the user can review the file structure and approve / reject."""
            pass



        graph.add_node("query_extractor", query_extractor)
        graph.add_node("structure_analyzer", structure_analyzer)
        graph.add_node("file_reader", file_reader)
        graph.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        graph.add_node("human_review", human_review)

        graph.add_edge(START, "query_extractor")
        graph.add_conditional_edges("query_extractor", self._should_analyze)
        graph.add_conditional_edges(
            "structure_analyzer", self._route_after_structure_analyzer
        )
        graph.add_edge("tools", "structure_analyzer")
        graph.add_conditional_edges(
            "human_review", self._route_after_human_review
        )
        graph.add_edge("file_reader", END)

        return graph.compile(
            interrupt_before=["human_review"],
            checkpointer=self.memory,
        )

    def new_thread_config(self) -> dict:
        """Create a fresh thread config (unique thread_id)."""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def get_state(self, thread_config: dict):
        """Get the current state snapshot from the checkpointer."""
        return self.compiled_graph.get_state(thread_config)

    def update_state(self, thread_config: dict, values: dict, as_node: str):
        """Update the checkpointed state as if *as_node* produced *values*."""
        return self.compiled_graph.update_state(
            thread_config, values, as_node=as_node
        )
