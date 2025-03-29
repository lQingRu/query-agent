from pydantic import BaseModel, Field
from agents.state import InitialState
from config.model import LargeLanguageModel, llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class QuestionStructureEval(BaseModel):
    score: int = Field(
        description="Numerical score representing quality of the question's structure",
        min=0,
        max=5,
    )
    reasoning: str = Field(description="Short reason for the score")
    proposal: str = Field(
        description="Suggested improvements, including areas to fix or improve the question",
        examples=["Reduce ambiguity", "Simplify language"],
    )


def question_structure_eval_agent(state: InitialState):
    PROMPT_TEMPLATE = """
    You are an expert assistant tasked with evaluating user queries for their suitability in semantic search. Given the query, assess it based on the following criteria:

    1. Clarity and Precision: Is the query clear and focused, without ambiguity or unnecessary complexity? Does it use specific keywords that will improve retrieval accuracy?

    2. Focus on a Single Concept: Does the query target a single, clear concept or topic, avoiding mixed or unrelated ideas that could confuse search results?

    3. Open-ended vs. Closed-ended: Is the query formatted appropriately for the expected answer (e.g., open-ended for exploratory searches, closed-ended for factual searches)?

    4. Temporal Terms: Does the query contain temporal words (e.g., "when", "recent", "past") that can decrease the precision of search results?

    5. Conciseness: Does the query avoid unnecessary words or convoluted phrasing that might hinder understanding or search performance?
    {format_instructions}

    **User's Question**
    {question}
    """
    model = llm(model=LargeLanguageModel.PHI_4)
    parser = JsonOutputParser(pydantic_object=QuestionStructureEval)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    response: QuestionStructureEval = chain.invoke({"question": state.question})
    print("[results] question_structure: ")
    print(response)
    return {"question_structure_eval": response}
