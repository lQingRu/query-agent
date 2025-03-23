from typing import List
from pydantic import BaseModel, Field
from agents.state import OverallState
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class RefinedQuestions(BaseModel):
    refined_questions: List[str] = Field(
        description="A list of questions that are refined based on the feedback",
        default=[],
    )


def refine_question_eval(state: OverallState):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with refining a user query for better semantic search performance. 
    
    The original query may contain issues that need improvement in terms of structure, keywords, abbreviations, and domain-specific terms. Given the evaluation feedback for the following factors, provide up to 5 refined versions of the original question. 
    
    Ensure that the refined questions maintain the original intent of the user's query, while improving clarity, semantic relevance, and search effectiveness. The refined questions should address the identified issues and be better for semantic search.
    
    {format_instructions}

    **User's Question**
    {question}
    
    **Feedback**
    {feedback}
    """
    model = llm(model=LargeLanguageModel.PHI_4)
    parser = JsonOutputParser(pydantic_object=RefinedQuestions)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question", "feedback"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: RefinedQuestions = chain.invoke(
        {
            "question": state.original_question,
            "feedback": str(state.evaluation_results.model_dump_json),
        }
    )
    print("[results] refine_question: ")
    print(response)

    _obj = RefinedQuestions.model_validate(response)
    # TODO: probably to use Send () to pass each refined question to evaluation
    return {"refined_questions": _obj.refined_questions}
