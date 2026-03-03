from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFacePipeline


class Models:
    _local_llm_instance = None
    
    @staticmethod
    def get_openai_model(model_name: str, temperature: float = 0.0):
        return ChatOpenAI(model=model_name, temperature=temperature)
    
    # @staticmethod
    # def get_anthropic_model(model_name: str, temperature: float = 0.0):
    #     return ChatAnthropic(model=model_name, temperature=temperature)