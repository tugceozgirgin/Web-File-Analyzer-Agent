from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    
    def __init__(self, name: str, tools: list, model_name: str):
        self.name = name
        self.tools = tools
        self.model_name = model_name

    @abstractmethod
    def forward(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward(state)
