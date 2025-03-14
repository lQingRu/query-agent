from typing import List
from pydantic import BaseModel, Field
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class RefinerOutput(BaseModel):
    class ScoreReasonMetric(BaseModel):
        score: int = Field(description="Score for metric", min=0, max=5)
        reason: str = Field(description="Short reason for the score")

    question_structure: ScoreReasonMetric
    abbreviation: ScoreReasonMetric
    domain_specific_term: ScoreReasonMetric
    keyword: ScoreReasonMetric
    open_ended_closed_ended: ScoreReasonMetric
    refined_questions: List[str]


def refine_question(question: str):
    """Evaluate a given question and return a score for each metric and propose refined questions

    Args:
        question (str): _description_
    """
    REFINE_PROMPT_TEMPLATE = """
    You are an expert assistant helping refine user queries for an AI-powered Retrieval-Augmented Generation (RAG) system.
    Your goal is to improve the effectiveness of retrieval by generating a clearer, more structured, and contextually enriched version of the input query that is best suited for semantic search.

    **Metrics**
    1. Question Structure: Ensure clarity and specificity.
    2. Abbreviations: Expand to improve understanding.
    3. Domain-Specific Terminology: Use precise terms relevant to the topic.
    4. Keywords: Emphasize essential terms for better retrieval.
    5. Open vs. Closed-Ended: Adjust based on the retrieval goal.

    **Task**
    1. Provide a score from 0 to 5 and a short reason of the score for each of the metric.
    2. If the given question requires refinement, propose 5 refined questions after taking into consideration of the factors. Do not change the intention of the user's question.
    {format_instructions}

    **Question**
    {question}
    """

    model = llm(model=LargeLanguageModel.PHI_4)
    parser = JsonOutputParser(pydantic_object=RefinerOutput)
    prompt = PromptTemplate(
        template=REFINE_PROMPT_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: RefinerOutput = chain.invoke({"question": question})
    print("REFINEMENT: ")
    print(response)
    return response


def refine_question_with_feedback(question: str, feedback: str):
    REFINE_FEEDBACK_PROMPT_TEMPLATE = """
    You are refining a set of queries for an AI-powered Retrieval-Augmented Generation (RAG) system, to improve the effectiveness of retrieval by generating a clearer, more structured, and contextually enriched version of the input query that is best suited for semantic search.
    Your task is to analyze the given feedback and adjust the queries accordingly while preserving the original intent.

    **Metrics**
    1. Question Structure: Ensure clarity and specificity.
    2. Abbreviations: Expand to improve understanding.
    3. Domain-Specific Terminology: Use precise terms relevant to the topic.
    4. Keywords: Emphasize essential terms for better retrieval.
    5. Open vs. Closed-Ended: Adjust based on the retrieval goal.

    **Original Question**
    {question}
    
    **Feedback on previous refined queries**
    {feedback}
    
     **Task**
    1. Generate 5 further improved queries that incorporate the feedback while enhancing retrieval quality.
    2. Provide a score from 0 to 5 and a short reason of the score for each of the metric.
    {format_instructions}
    """

    model = llm(model=LargeLanguageModel.PHI_4)
    parser = JsonOutputParser(pydantic_object=RefinerOutput)
    prompt = PromptTemplate(
        template=REFINE_FEEDBACK_PROMPT_TEMPLATE,
        input_variables=["question", "feedback"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: RefinerOutput = chain.invoke({"question": question, "feedback": feedback})
    print(response)
    return response


# def refine_question(question: str, feedback: str):
#     """Evaluate a given question and return a score for each metric and propose refined questions

#     Args:
#         question (str): _description_
#     """
#     REFINE_PROMPT_TEMPLATE = """
#     You are an expert in evaluating and refining questions for the purpose of semantic search.

#     **Metrics**
#     1. Question Structure: Ensure clarity and specificity.
#     2. Abbreviations: Expand to improve understanding.
#     3. Domain-Specific Terminology: Use precise terms relevant to the topic.
#     4. Keywords: Emphasize essential terms for better retrieval.
#     5. Open vs. Closed-Ended: Adjust based on the retrieval goal.

#     **Task**
#     1. Provide a score from 0 to 5 and a short reason of the score for each of the metric.
#     2. If the given question requires refinement, propose 5 refined questions after taking into consideration of the factors. Do not change the intention of the user's question.
#     """
#     prompt = ChatPromptTemplate(
#         messages=[
#             ("system", REFINE_PROMPT_TEMPLATE),
#             ("placeholder", "{conversation}"),
#         ],
#         output_parser=JsonOutputParser(pydantic_object=RefinerOutput),
#     )

#     model = llm(model=LargeLanguageModel.PHI_4)

#     chain = prompt | model

#     # After reflection
#     if feedback:
#         conversation = [("human", f"{feedback}")]
#         response: RefinerOutput = chain.invoke({"conversation": conversation})
#     else:
#         conversation = [("human", f"{question}")]
#         response: RefinerOutput = chain.invoke({"conversation": conversation})
#     print(response.model_dump())
#     return response
