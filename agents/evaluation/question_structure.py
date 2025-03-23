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
    You are an expert assistance in helping to refine user queries so that the query is best crafted for a precise semantic search.
    
    Given the query, evaluate its structure for clarity, coherence, grammatical correctness, and focus, considering the following factors:

    Clarity and Precision: Is the query clear and precise without ambiguity?

    Focus on Key Information: Is the question concise and focused on the main topic, avoiding irrelevant details?

    Context Appropriateness: Does the query match the expected context (e.g., technical vs. general)?

    Grammatical Correctness and Syntax: Are there any grammatical issues or awkward sentence structures that might reduce clarity?

    Open-ended vs. Closed-ended Question: Is the question type appropriate for the expected answer (exploratory vs. fact-based)?

    Conciseness: Does the query contain unnecessary words or convoluted phrasing that could affect understanding?

    Tone and User Intent: Does the tone of the query align with the user's likely intent (e.g., informational, instructional)?
    
    {format_instructions}

    **User's Question**
    {question}
    """
    model = llm(model=LargeLanguageModel.PHI_4)
    parser = JsonOutputParser(pydantic_object=QuestionStructureEval)
    # structured_llm = model.with_structured_output(QuestionStructureEval)

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

    # question_structure: QuestionStructureEval = {
    #     "proposal": "Updates on NewWater in Singapore",
    #     "reasoning": "Clarify logical flow",
    #     "score": 3,
    # }
    # return {"question_structure_eval": question_structure}
