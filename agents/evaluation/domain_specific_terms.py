from typing import List
from pydantic import BaseModel, Field
from agents.state import InitialState
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class DomainSpecificTermEval(BaseModel):
    domain_terms: List[str] = Field(
        description="A list of domain-specific terms present in the query", default=[]
    )
    proposal: str = Field(
        description="A suggestion on how to handle these terms",
        default="",
        examples=[
            "Replace these domain terms with synonyms",
            "Provide explanatory context",
            "Extract these domain terms as keywords",
        ],
    )


def domain_specific_terms_eval_agent(state: InitialState):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with refining user queries to ensure they are optimized for precise semantic search. 
    
    Given the query, evaluate whether it contains any domain-specific terms that may be difficult to understand or interpret without specialized knowledge of the domain. If such terms are present, list them and suggest ways to address them, keeping in mind that the goal is to ensure the query will work effectively in semantic search, where the embedding model will accurately represent the query with vectors.
    
    {format_instructions}

    **User's Question**
    {question}
    """
    model = llm(model=LargeLanguageModel.GEMMA_3_1b)
    parser = JsonOutputParser(pydantic_object=DomainSpecificTermEval)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: DomainSpecificTermEval = chain.invoke({"question": state.question})
    print("[results] domain_specific_terms: ")
    print(response)
    return {"domain_specific_term_eval": response}
