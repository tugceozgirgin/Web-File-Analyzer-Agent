from typing import TypedDict, Any, List, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic import Field
from enum import Enum
from datetime import datetime

class FileTypes(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"

class File(BaseModel):
    file_name: str
    file_type: FileTypes
    file_url: str

class FileStructure(BaseModel):
    title: Optional[str] =Field(default=None, description="The title of the files.")
    file_names:Optional[List[File]] =Field(default=None, description="The files under the title.")

class FileStructureList(BaseModel):
    """Wrapper used as the structured-output schema for the LLM."""
    page_title: Optional[str] = Field(
        default=None,
        description="The title of the web page (extracted from page content or metadata).",
    )
    page_description: Optional[str] = Field(
        default=None,
        description="A short one-sentence description of what the page contains.",
    )
    file_structures: List[FileStructure] = Field(
        default_factory=list,
        description="List of file groups extracted from the page.",
    )

class Filters(BaseModel):
    url: Optional[str] =Field(default=None, description="The URL of the website will be used to search for files.")
    file_type: Optional[List[FileTypes]]=Field(default=None, description="The type of files to search for.")
    categories: Optional[List[str]]=Field(default=None, description="The categories of the files to search for.")
    start_date: Optional[datetime]=Field(default=None, description="The start date of the relevant files to search for.")
    end_date: Optional[datetime]=Field(default=None, description="The end date of the relevant files to search for.")
    
class FileSummary(BaseModel):
    """Summary of a single file produced by the FileReaderAgent."""
    file_name: str = Field(description="Original file name.")
    file_type: FileTypes = Field(description="File format (pdf, docx, xlsx, csv).")
    file_url: str = Field(description="Download URL of the file.")
    summary: str = Field(default="", description="LLM-generated summary of the file content.")
    error: Optional[str] = Field(default=None, description="Error message if the file could not be processed.")

class WebFileAnalyzerState(TypedDict, total=False):
    input: str
    filters: Filters
    file_structures: List[FileStructure]
    page_title: Optional[str]
    page_description: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    file_structure_summary: Optional[str]
    human_approval: Optional[str] 
    human_feedback: Optional[str] 
    page_content: Optional[str] 
    file_summaries: Optional[List[FileSummary]]
    output: Optional[str]