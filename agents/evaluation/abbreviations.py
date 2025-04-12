from typing import List
from pydantic import BaseModel, Field
from agents.state import InitialState
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException


class AbbreviationEval(BaseModel):
    class AbbreviationExpansion(BaseModel):
        abbreviation: str = Field(description="Extracted abbreviation from question")
        expansion: List[str] = Field(
            description="Proposed expanded terms for abbreviation", default=[]
        )

    abbreviations: List[AbbreviationExpansion] = Field(
        description="List of abbreviations and their expanded terms", default=[]
    )


def abbreviations_eval(state: InitialState):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with refining user queries for optimal performance in hybrid search. Given the query, identify any abbreviations and expand them into their full forms or common alternative expressions.

    The goal is to ensure the query is clear and properly represented for semantic search. Expanding abbreviations will help the embedding model accurately interpret and represent the query in vector form, leading to improved search results.

    Ensure that all terms are correctly expanded and represented for optimal semantic search performance.
    
    {format_instructions}
    
    # User's Question
    {question}
    """
    model = llm(model=LargeLanguageModel.LLAMA_3_GROQ_TOOL_USE)
    parser = JsonOutputParser(pydantic_object=AbbreviationEval)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    try:
        response: AbbreviationEval = chain.invoke({"question": state.question})
        print("[results] abbreviations_eval: ")
        print(response)
        return {"abbreviations_eval": response}
    except OutputParserException as ex:
        print(ex)
        return {"abbreviations_eval": AbbreviationEval(abbreviations=[]).model_dump()}
