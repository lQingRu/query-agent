from typing import List
from pydantic import BaseModel, Field
from agents.state import InitialState
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class KeywordEval(BaseModel):
    class KeywordExpansion(BaseModel):
        keyword: str = Field(description="Extracted keyword from question")
        expanded_keywords: List[str] = Field(
            description="Fuzzed and expanded variants of keyword", default=[]
        )

    keywords: List[KeywordExpansion] = Field(
        description="List of keywords and their expansions", default=[]
    )


def keywords_eval_agent(state: InitialState):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with refining user queries for optimal performance in hybrid search. Given the query, extract the most relevant keywords that represent the core concepts of the query.

    For each keyword, apply fuzzing and expansion techniques to improve search recall. This includes:

    Synonym expansion: Identifying potential synonyms for the extracted keywords.

    Language fuzzing: If the query contains terms in non-English languages (e.g., Indonesian, Spanish), fuzz and translate the keywords into English equivalents where appropriate.

    Related term expansion: Expanding the keyword set with related terms or phrases that convey the same or similar meaning.

    The goal is to ensure that the keywords used in the BM25 search part of the hybrid search ensure higher recall and better results without changing the intention of the question.
    
    {format_instructions}

    **User's Question**
    {question}
    """
    model = llm(model=LargeLanguageModel.LLAMA_3_GROQ_TOOL_USE)
    parser = JsonOutputParser(pydantic_object=KeywordEval)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: KeywordEval = chain.invoke({"question": state.question})
    print("[results] keywords_eval: ")
    print(response)
    return {"keywords_eval": response}
