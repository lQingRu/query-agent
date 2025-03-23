from typing import List
from pydantic import BaseModel, Field
from agents.state import InitialState
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class AbbreviationEval(BaseModel):
    class AbbreviationExpansion(BaseModel):
        abbreviation: str = Field(description="Extracted abbreviation from question")
        expansion: List[str] = Field(
            description="Proposed expanded terms for abbreviation", default=[]
        )

    abbreviations: AbbreviationExpansion = Field(
        description="Abreviation expansion list", default=[]
    )


def abbreviations_eval(state: InitialState):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with refining user queries for optimal performance in hybrid search. Given the query, identify any abbreviations and expand them into their full forms or alternative common expressions. 
    
    The goal is to ensure that the query is clear and properly represented for semantic search. Specifically, make sure that the full forms of the abbreviations are included, as this will help the embedding model to accurately represent the query in vector form. 
    
    This will improve the search results by ensuring all terms are correctly interpreted and represented in the semantic search space.
    
    {format_instructions}

    **User's Question**
    {question}
    """
    model = llm(model=LargeLanguageModel.GEMMA_3_1b)
    parser = JsonOutputParser(pydantic_object=AbbreviationEval)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: AbbreviationEval = chain.invoke({"question": state.question})
    print("[results] abbreviations_eval: ")
    print(response)
    return {"abbreviations_eval": response}
