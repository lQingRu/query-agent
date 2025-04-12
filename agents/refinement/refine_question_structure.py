from typing import List
from pydantic import BaseModel, Field
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from agents.state import RefineQueryStructure


class RefinedQuestions(BaseModel):
    refined_questions: List[str] = Field(
        description="A list of questions that are refined based on the feedback",
        default=[],
    )


def refine_question_structure_node(state: RefineQueryStructure):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with refining user queries to enhance semantic search performance.

    Based on the proposed improvements, generate up to five refined versions of the original query. Ensure that each refinement:
    - Maintains the original intent
    - Improves clarity, semantic relevance, and search effectiveness
    - Incorporates feedback to enhance query quality
    
    {format_instructions}
    
    # Proposed Improvements
    {proposal}
    
    # User's Question
    {question}
    """
    model = llm(model=LargeLanguageModel.LLAMA_3_GROQ_TOOL_USE)
    parser = JsonOutputParser(pydantic_object=RefinedQuestions)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question", "proposal"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: RefinedQuestions = chain.invoke(
        {
            "question": state.question,
            "proposal": str(state.question_structure_eval.proposal),
        }
    )
    print("[results] refine_question_structure_eval: ")
    print(response)

    _obj = RefinedQuestions.model_validate(response)
    return {"refined_questions": _obj.refined_questions}
