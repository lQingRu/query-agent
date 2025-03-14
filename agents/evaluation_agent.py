from typing import Dict, List
from pydantic import BaseModel, Field
from config.model import LargeLanguageModel, llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class EvaluatorOutput(BaseModel):
    class ScoreReasonMetric(BaseModel):
        score: int = Field(description="Score for metric", min=0, max=5)
        reason: str = Field(description="Short reason for the score")

    eliminated_refined_questions: Dict[str, ScoreReasonMetric] = Field(
        description="Key is the refined question and value is the score and reason of the score"
    )
    best_refined_question: Dict[str, ScoreReasonMetric] = Field(
        description="Key is the best refined question and value is the score and reason of the score"
    )


def evaluate_questions(question: str, refined_questions: List[str]):
    """Evaluate the proposed questions with the original question and return the score of each question"""
    EVALUATION_PROMPT_TEMPLATE = """
    You are an evaluation agent responsible for assessing the quality of refined queries for semantic search. Your goal is to determine which refined query is most effective for retrieving relevant and high-quality information.

    For each refined query, assess the following:
    1. User Intention: Does the refined query preserve the intent of the original query while improving clarity?
    2. Retrieval Effectiveness: Would this query improve search results by making it more specific, structured, or relevant?
    3. Query Expansion Quality: If expanded, does it add meaningful context without introducing irrelevant details?
    4. Query Decomposition: If split into subqueries, do they comprehensively cover the original intent?
    5. Embedding Distance: Compare the embeddings of the original query and the refined query to ensure a meaningful difference.

    **Tasks**
    1. Compare each refined query against the original based on the evaluation criteria.
    2. Assign a score (0-5) and a short reasoning to each refined query based on how well it improves search effectiveness.
    3. Select the best refined query and justify why it performs the best.
    {format_instructions}
    
    **Original User Question**
    {question}
    
    **Refined Questions**
    {refined_questions}
    """

    refined_questions_formatted = ""
    for i, question in enumerate(refined_questions):
        refined_questions_formatted += f"{i+1}. {question}\n"
    model = llm(model=LargeLanguageModel.PHI_4)
    parser = JsonOutputParser(pydantic_object=EvaluatorOutput)
    prompt = PromptTemplate(
        template=EVALUATION_PROMPT_TEMPLATE,
        input_variables=["question", "refined_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: EvaluatorOutput = chain.invoke(
        {"question": question, "refined_questions": refined_questions_formatted}
    )
    print("EVALUATION:")
    print(response)
    return response
